use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use common::bitvec::{BitSliceExt as _, BitVec};
use common::counter::hardware_counter::HardwareCounterCell;
use common::progress_tracker::ProgressTracker;
use common::types::{DeferredBehavior, PointOffsetType};
use log::{debug, trace};
use rand::Rng;
use rayon::ThreadPool;
use rayon::prelude::*;

use crate::common::operation_error::{OperationError, OperationResult, check_process_stopped};
use crate::id_tracker::{IdTrackerEnum, IdTrackerRead};
use crate::index::PayloadIndexRead;
use crate::index::condition_checker::ConditionCheckerEnum;
use crate::index::field_index::PayloadBlockCondition;
use crate::index::hnsw_index::HnswM;
use crate::index::hnsw_index::build_condition_checker::BuildConditionChecker;
use crate::index::hnsw_index::config::HnswGraphConfig;
use crate::index::hnsw_index::gpu::gpu_insert_context::GpuInsertContext;
use crate::index::hnsw_index::graph_layers_builder::GraphLayersBuilder;
use crate::index::hnsw_index::hnsw::{
    HNSW_BUILD_MAX_PAR_LEN, HNSW_USE_HEURISTIC, SINGLE_THREADED_HNSW_BUILD_THRESHOLD,
};
use crate::index::hnsw_index::point_scorer::{
    BlockVectors, FilteredScorer, MAX_BLOCK_GATHER_BYTES,
};
use crate::index::query_optimization::optimized_filter::OptimizedFilter;
use crate::index::struct_payload_index::StructPayloadIndex;
use crate::index::visited_pool::{VisitedListHandle, VisitedPool};
use crate::json_path::JsonPath;
use crate::types::Condition::Field;
use crate::types::{FieldCondition, Filter};
use crate::vector_storage::quantized::quantized_vectors::QuantizedVectors;
use crate::vector_storage::{VectorStorageEnum, VectorStorageRead};

/// Fields that get additional HNSW links, each paired with its progress subtask under the
/// returned `additional_links` tracker. `None` when there is nothing to build.
pub(super) fn additional_links_fields(
    payload_index: &StructPayloadIndex,
    payload_m: HnswM,
    progress: &ProgressTracker,
) -> Option<(ProgressTracker, Vec<(ProgressTracker, JsonPath)>)> {
    if payload_m.m == 0 {
        return None;
    }
    let fields = payload_index.with_view(|v| v.indexed_fields());
    if fields.is_empty() {
        return None;
    }
    let progress_additional_links = progress.subtask("additional_links");
    let fields = fields
        .into_iter()
        .filter_map(|(field, payload_schema)| {
            let subtask_name = format!("{}:{field}", payload_schema.name());
            if payload_schema.enable_hnsw() {
                Some((progress_additional_links.subtask(subtask_name), field))
            } else {
                debug!("enable_hnsw=false. Skip building additional index for field {field}");
                None
            }
        })
        .collect::<Vec<_>>();
    Some((progress_additional_links, fields))
}

/// Build per-payload-block subgraphs for every field in `indexed_fields` and merge them
/// into `graph_layers_builder`.
///
/// Returns the number of vectors that got indexed through these subgraphs.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_additional_links<R: Rng + ?Sized>(
    id_tracker: &IdTrackerEnum,
    vector_storage: &VectorStorageEnum,
    quantized_vectors: &Option<QuantizedVectors>,
    payload_index: &StructPayloadIndex,
    gpu_insert_context: &mut Option<GpuInsertContext<'_>>,
    graph_layers_builder: &mut GraphLayersBuilder,
    config: &HnswGraphConfig,
    payload_m: HnswM,
    indexed_fields: Vec<(ProgressTracker, JsonPath)>,
    progress_additional_links: ProgressTracker,
    pool: &ThreadPool,
    rng: &mut R,
    stopped: &AtomicBool,
) -> OperationResult<usize> {
    let total_vector_count = vector_storage.total_vector_count();
    let deleted_bitslice = vector_storage.deleted_vector_bitslice();

    progress_additional_links.start();

    // Calculate true average number of links per vertex in the HNSW graph
    // to better estimate percolation threshold
    let average_links_per_0_level = graph_layers_builder.get_average_connectivity_on_level(0);
    let average_links_per_0_level_int = (average_links_per_0_level as usize).max(1);

    // Estimate connectivity of the main graph
    let all_points = id_tracker
        .point_mappings()
        .iter_internal_excluding(deleted_bitslice)
        .collect::<Vec<_>>();

    // According to percolation theory, random graph becomes disconnected
    // if 1/K points are left, where K is average number of links per point
    // So we need to sample connectivity relative to this bifurcation point, but
    // not exactly at 1/K, as at this point graph is very sensitive to noise.
    //
    // Instead, we choose sampling point at 2/K, which expects graph to still be
    // mostly connected, but still have some measurable disconnected components.

    let percolation = 1. - 2. / (average_links_per_0_level_int as f32);

    let required_connectivity = if average_links_per_0_level_int >= 4 {
        let global_graph_connectivity = [
            graph_layers_builder.subgraph_connectivity(rng, &all_points, percolation),
            graph_layers_builder.subgraph_connectivity(rng, &all_points, percolation),
            graph_layers_builder.subgraph_connectivity(rng, &all_points, percolation),
        ];

        debug!("graph connectivity: {global_graph_connectivity:?} @ {percolation}");

        global_graph_connectivity
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
    } else {
        // Main graph is too small to estimate connectivity,
        // we can't shortcut sub-graph building
        None
    };

    let mut indexed_vectors_set = if config.m != 0 {
        // Every vector is already indexed in the main graph, so skip counting.
        BitVec::new()
    } else {
        BitVec::repeat(false, total_vector_count)
    };

    // On CPU every block is built in its own id space (see `build_block_compact`). The GPU
    // builder needs a segment-sized builder, so it keeps the segment-wide path below.
    let use_compact_blocks = gpu_insert_context.is_none() && !test_hooks::force_legacy();

    let visited_pool = VisitedPool::new();
    let mut block_filter_list = visited_pool.get(total_vector_count);

    // One builder serves every block. It is sized for the whole segment, so allocating it
    // per block, and merging it back by scanning every point, would cost O(segment) on a
    // single thread for each block. `merge_block_from` moves out only the block's links and
    // leaves the builder empty for the next block.
    let mut block_graph: Option<GraphLayersBuilder> = None;

    for (index_pos, (field_progress, field)) in indexed_fields.into_iter().enumerate() {
        field_progress.start();

        debug!("building additional index for field {field}");

        let is_tenant = payload_index.is_tenant(&field);

        // It is expected, that graph will become disconnected less than
        // $1/m$ points left.
        // So blocks larger than $1/m$ are not needed.
        // We add multiplier for the extra safety.
        const PERCOLATION_MULTIPLIER: usize = 4;
        let max_block_size = if config.m > 0 {
            total_vector_count / average_links_per_0_level_int * PERCOLATION_MULTIPLIER
        } else {
            usize::MAX
        };

        let counter = field_progress.track_progress(None);

        let mut process_block = |payload_block: PayloadBlockCondition| {
            check_process_stopped(stopped)?;

            if payload_block.cardinality > max_block_size {
                return Ok(());
            }

            let points_to_index = condition_points(
                payload_block.condition,
                payload_index,
                vector_storage,
                stopped,
            )?;

            // This is a heuristic to skip building graph for mostly deleted blocks.
            // It might be, that majority of points do not actually have vectors
            // (vectors marked as deleted), so we can avoid building graph for such blocks.
            //
            // FYI: query heuristic does account
            // for deleted vectors via [`adjust_to_available_vectors`]
            const DELETED_POINTS_FACTOR: usize = 4; // allow block to have up to 75% of deleted points and still be indexed

            if points_to_index.len() <= config.full_scan_threshold / DELETED_POINTS_FACTOR {
                return Ok(());
            }

            if !is_tenant
                && index_pos > 0
                && let Some(required_connectivity) = required_connectivity
            {
                // Always build for tenants
                let graph_connectivity =
                    graph_layers_builder.subgraph_connectivity(rng, &points_to_index, percolation);

                if graph_connectivity >= required_connectivity {
                    trace!(
                        "skip building additional HNSW links for {field}, connectivity {graph_connectivity:.4} >= {required_connectivity:.4}"
                    );
                    return Ok(());
                }
                trace!("graph connectivity: {graph_connectivity} for {field}");
            }

            test_hooks::count_built();

            if use_compact_blocks {
                build_block_compact(
                    &points_to_index,
                    graph_layers_builder,
                    pool,
                    id_tracker,
                    vector_storage,
                    quantized_vectors,
                    payload_m,
                    config.ef_construct,
                    stopped,
                    &counter,
                )?;
                if !indexed_vectors_set.is_empty() {
                    for &point_id in &points_to_index {
                        indexed_vectors_set.set(point_id as usize, true);
                    }
                }
                return Ok(());
            }

            let additional_graph = block_graph.get_or_insert_with(|| {
                GraphLayersBuilder::new_with_params(
                    total_vector_count,
                    payload_m,
                    config.ef_construct,
                    1,
                    HNSW_USE_HEURISTIC,
                    false,
                )
            });

            let gpu_constructed_graph = build_filtered_graph(
                id_tracker,
                vector_storage,
                quantized_vectors,
                gpu_insert_context,
                payload_index,
                pool,
                stopped,
                additional_graph,
                &points_to_index,
                &mut block_filter_list,
                &mut indexed_vectors_set,
                &counter,
            )?;
            match gpu_constructed_graph {
                Some(gpu_graph) => graph_layers_builder.merge_from_other(gpu_graph),
                None => graph_layers_builder.merge_block_from(additional_graph, &points_to_index),
            }
            Ok(())
        };

        payload_index.with_view(|v| {
            v.for_each_payload_block(&field, config.full_scan_threshold, &mut process_block)
        })?;
    }
    Ok(indexed_vectors_set.count_ones())
}

/// Build one payload block in its own id space and merge it into `graph`.
///
/// Points are numbered `0..points.len()` in `points` order, so the builder, its visited lists
/// and its links are sized to the block, not the segment, and every candidate a search can
/// reach is a block member: no filter check is needed to keep the search inside the block.
/// When the block's vectors fit in [`MAX_BLOCK_GATHER_BYTES`], they are first copied into one
/// contiguous buffer and scored from there. The copy holds the same bytes the storage would
/// return, so the graph is the same one the segment-wide path builds.
#[allow(clippy::too_many_arguments)]
fn build_block_compact(
    points: &[PointOffsetType],
    graph: &GraphLayersBuilder,
    pool: &ThreadPool,
    id_tracker: &IdTrackerEnum,
    vector_storage: &VectorStorageEnum,
    quantized_vectors: &Option<QuantizedVectors>,
    payload_m: HnswM,
    ef_construct: usize,
    stopped: &AtomicBool,
    counter: &AtomicU64,
) -> OperationResult<()> {
    let vec_deleted = vector_storage.deleted_vector_bitslice();
    let point_deleted = id_tracker.deleted_point_bitslice();
    let mut block_deleted = BitVec::repeat(false, points.len());
    for (local_id, &point_id) in points.iter().enumerate() {
        let deleted = vec_deleted.get_bit(point_id as usize).unwrap_or(false)
            || point_deleted.get_bit(point_id as usize).unwrap_or(false);
        if deleted {
            block_deleted.set(local_id, true);
        }
    }

    let block_vectors = BlockVectors::try_gather(
        points,
        vector_storage,
        quantized_vectors.as_ref(),
        MAX_BLOCK_GATHER_BYTES,
    );
    if block_vectors.is_some() {
        test_hooks::count_gathered();
    }

    let block_graph = GraphLayersBuilder::new_with_params(
        points.len(),
        payload_m,
        ef_construct,
        1,
        HNSW_USE_HEURISTIC,
        false,
    );

    let insert_point = |local_id: PointOffsetType| {
        check_process_stopped(stopped)?;

        // This hardware counter can be discarded, since it is only used for internal operations
        let internal_hardware_counter = HardwareCounterCell::disposable();

        let points_scorer = FilteredScorer::new_block_scorer(
            points[local_id as usize],
            points,
            vector_storage,
            quantized_vectors.as_ref(),
            block_vectors.as_ref(),
            &block_deleted,
            internal_hardware_counter,
        )?;

        block_graph.link_new_point(local_id, points_scorer);

        counter.fetch_add(1, Ordering::Relaxed);

        Ok::<_, OperationError>(())
    };

    let block_len = points.len() as PointOffsetType;
    let first_points = block_len.min(SINGLE_THREADED_HNSW_BUILD_THRESHOLD as PointOffsetType);

    // First index points in single thread so ensure warm start for parallel indexing process
    for local_id in 0..first_points {
        insert_point(local_id)?;
    }
    // Once initial structure is built, index remaining points in parallel
    if block_len > first_points {
        pool.install(|| {
            (first_points..block_len)
                .into_par_iter()
                .with_max_len(HNSW_BUILD_MAX_PAR_LEN)
                .try_for_each(insert_point)
        })?;
    }

    graph.merge_block(block_graph, points);
    Ok(())
}

/// Switches and counters for tests that compare the compact and segment-wide block paths.
/// The block loop runs on the thread that called the build, so thread-locals are enough.
#[cfg(test)]
pub(crate) mod test_hooks {
    use std::cell::Cell;

    thread_local! {
        static FORCE_LEGACY: Cell<bool> = const { Cell::new(false) };
        static BUILT: Cell<usize> = const { Cell::new(0) };
        static GATHERED: Cell<usize> = const { Cell::new(0) };
        static SNAPSHOT: std::cell::RefCell<Option<GraphSnapshot>> =
            const { std::cell::RefCell::new(None) };
    }

    /// Every point's links per level in stored order, plus the entry points.
    pub type GraphSnapshot = (Vec<Vec<Vec<common::types::PointOffsetType>>>, String);

    /// Record the finished graph of a build on this thread, before it is serialized.
    pub(crate) fn record_graph(
        builder: &crate::index::hnsw_index::graph_layers_builder::GraphLayersBuilder,
    ) {
        let snapshot = (
            builder.links_snapshot(),
            format!("{:?}", *builder.get_entry_points()),
        );
        SNAPSHOT.set(Some(snapshot));
    }

    pub fn take_graph() -> Option<GraphSnapshot> {
        SNAPSHOT.take()
    }

    /// Build blocks with the segment-wide path on this thread.
    pub fn set_force_legacy(value: bool) {
        FORCE_LEGACY.set(value);
    }

    pub(super) fn force_legacy() -> bool {
        FORCE_LEGACY.get()
    }

    pub(super) fn count_built() {
        BUILT.set(BUILT.get() + 1);
    }

    pub(super) fn count_gathered() {
        GATHERED.set(GATHERED.get() + 1);
    }

    /// (blocks built, blocks gathered) on this thread since the last call.
    pub fn take_counts() -> (usize, usize) {
        (BUILT.replace(0), GATHERED.replace(0))
    }
}

#[cfg(not(test))]
mod test_hooks {
    pub(super) fn force_legacy() -> bool {
        false
    }
    pub(super) fn count_built() {}
    pub(super) fn count_gathered() {}
}

/// Get list of points for indexing, associated with payload block filtering condition
fn condition_points(
    condition: FieldCondition,
    payload_index: &StructPayloadIndex,
    vector_storage: &VectorStorageEnum,
    stopped: &AtomicBool,
) -> OperationResult<Vec<PointOffsetType>> {
    let filter = Filter::new_must(Field(condition));

    let disposed_hw_counter = HardwareCounterCell::disposable(); // Internal operation. No measurements needed

    let deleted_bitslice = vector_storage.deleted_vector_bitslice();

    payload_index.with_view(|v| {
        let cardinality_estimation = v.estimate_cardinality(&filter, &disposed_hw_counter)?;
        Ok(v.iter_filtered_points(
            &filter,
            &cardinality_estimation,
            &disposed_hw_counter,
            stopped,
            DeferredBehavior::WithDeferred,
        )?
        .filter(|&point_id| !deleted_bitslice.get_bit(point_id as usize).unwrap_or(false))
        .collect())
    })
}

/// Insert `points_to_index` into `graph_layers_builder`, linking them only to each other.
///
/// Returns the graph instead when it was built on the GPU; `graph_layers_builder` is then
/// left untouched.
#[allow(clippy::too_many_arguments)]
#[allow(unused_variables)]
#[allow(clippy::needless_pass_by_ref_mut)]
fn build_filtered_graph(
    id_tracker: &IdTrackerEnum,
    vector_storage: &VectorStorageEnum,
    quantized_vectors: &Option<QuantizedVectors>,
    #[allow(unused_variables)] gpu_insert_context: &mut Option<GpuInsertContext<'_>>,
    payload_index: &StructPayloadIndex,
    pool: &ThreadPool,
    stopped: &AtomicBool,
    graph_layers_builder: &GraphLayersBuilder,
    points_to_index: &[PointOffsetType],
    block_filter_list: &mut VisitedListHandle,
    indexed_vectors_set: &mut BitVec,
    counter: &AtomicU64,
) -> OperationResult<Option<GraphLayersBuilder>> {
    block_filter_list.next_iteration();

    for block_point_id in points_to_index.iter().copied() {
        block_filter_list.check_and_update_visited(block_point_id);
        if !indexed_vectors_set.is_empty() {
            indexed_vectors_set.set(block_point_id as usize, true);
        }
    }

    #[cfg(feature = "gpu")]
    if let Some(gpu_constructed_graph) =
        crate::index::hnsw_index::hnsw::gpu_build::build_filtered_graph_on_gpu(
            id_tracker,
            vector_storage,
            quantized_vectors,
            gpu_insert_context.as_mut(),
            graph_layers_builder,
            block_filter_list,
            points_to_index,
            stopped,
        )?
    {
        return Ok(Some(gpu_constructed_graph));
    }

    let insert_points = |block_point_id| {
        check_process_stopped(stopped)?;

        // This hardware counter can be discarded, since it is only used for internal operations
        let internal_hardware_counter = HardwareCounterCell::disposable();

        let block_condition_checker =
            OptimizedFilter::from_checker(ConditionCheckerEnum::Build(BuildConditionChecker {
                filter_list: block_filter_list,
                current_point: block_point_id,
            }));
        let points_scorer = FilteredScorer::new_internal(
            block_point_id,
            vector_storage,
            quantized_vectors.as_ref(),
            Some(block_condition_checker),
            id_tracker.deleted_point_bitslice(),
            internal_hardware_counter,
        )?;

        graph_layers_builder.link_new_point(block_point_id, points_scorer);

        counter.fetch_add(1, Ordering::Relaxed);

        Ok::<_, OperationError>(())
    };

    let first_points = points_to_index
        .len()
        .min(SINGLE_THREADED_HNSW_BUILD_THRESHOLD);

    // First index points in single thread so ensure warm start for parallel indexing process
    for point_id in points_to_index[..first_points].iter().copied() {
        insert_points(point_id)?;
    }
    // Once initial structure is built, index remaining points in parallel
    // So that each thread will insert points in different parts of the graph,
    // it is less likely that they will compete for the same locks
    if points_to_index.len() > first_points {
        pool.install(|| {
            points_to_index[first_points..]
                .par_iter()
                .copied()
                .with_max_len(HNSW_BUILD_MAX_PAR_LEN)
                .try_for_each(insert_points)
        })?;
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    //! The compact block path must build exactly the graph of the segment-wide path.
    //! Adapted from the fixtures of #10046.

    // Config structs keep their deprecated placement fields until 2.0.
    #![allow(deprecated)]

    use std::sync::Arc;

    use atomic_refcell::AtomicRefCell;
    use common::budget::ResourcePermit;
    use common::flags::FeatureFlags;
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use rstest::rstest;
    use tempfile::Builder;

    use super::*;
    use crate::data_types::vectors::{DEFAULT_VECTOR_NAME, only_default_vector};
    use crate::entry::entry_point::SegmentEntry;
    use crate::fixtures::index_fixtures::random_vector;
    use crate::id_tracker::IdTracker;
    use crate::index::PayloadIndex;
    use crate::index::hnsw_index::hnsw::{HNSWIndex, HnswIndexOpenArgs};
    use crate::payload_json;
    use crate::segment::Segment;
    use crate::segment_constructor::VectorIndexBuildArgs;
    use crate::segment_constructor::simple_segment_constructor::build_simple_segment;
    use crate::types::{
        Distance, HnswConfig, HnswGlobalConfig, PayloadSchemaType, QuantizationConfig,
        SeqNumberType, TurboQuantBitSize, TurboQuantQuantizationConfig, TurboQuantization,
    };
    use crate::vector_storage::quantized::quantized_vectors::{
        QuantizedVectors, QuantizedVectorsStorageType,
    };

    const DIM: usize = 8;
    const NUM_VECTORS: u64 = 2_000;
    /// Every value covers 250 points, enough for its block to clear the full-scan threshold.
    const NUM_VALUES: u64 = 8;
    const SEED: u64 = 42;
    const KEY: &str = "field";

    #[derive(Clone, Copy, Debug)]
    enum Field {
        Keyword,
        /// Integer values produce overlapping range blocks.
        Integer,
        /// Every 9th point dropped from the id tracker while its vector stays live.
        KeywordWithStalePoints,
    }

    fn build_segment(dir: &std::path::Path, field: Field) -> Segment {
        let mut rng = StdRng::seed_from_u64(SEED);
        let hw_counter = HardwareCounterCell::new();
        let mut segment = build_simple_segment(dir, DIM, Distance::Cosine).unwrap();
        for n in 0..NUM_VECTORS {
            let value = n / (NUM_VECTORS / NUM_VALUES);
            let payload = match field {
                Field::Integer => payload_json! {KEY: value as i64},
                Field::Keyword | Field::KeywordWithStalePoints => {
                    payload_json! {KEY: format!("value-{value}")}
                }
            };
            let vector = random_vector(&mut rng, DIM);
            segment
                .upsert_point(
                    n as SeqNumberType,
                    n.into(),
                    only_default_vector(&vector),
                    &hw_counter,
                )
                .unwrap();
            segment
                .set_full_payload(n as SeqNumberType, n.into(), &payload, &hw_counter)
                .unwrap();
        }
        let schema = match field {
            Field::Integer => PayloadSchemaType::Integer,
            Field::Keyword | Field::KeywordWithStalePoints => PayloadSchemaType::Keyword,
        };
        segment
            .payload_index
            .borrow_mut()
            .set_indexed(&JsonPath::new(KEY), schema, &hw_counter)
            .unwrap();
        if let Field::KeywordWithStalePoints = field {
            let mut id_tracker = segment.id_tracker.borrow_mut();
            for n in (0..NUM_VECTORS).step_by(9) {
                id_tracker.drop(n.into()).unwrap();
            }
        }
        segment
    }

    /// Build the index on this thread and return (graph, blocks built, blocks gathered).
    fn build(
        segment: &Segment,
        legacy: bool,
        quantize: bool,
    ) -> (test_hooks::GraphSnapshot, usize, usize) {
        let stopped = AtomicBool::new(false);
        let hnsw_dir = Builder::new().prefix("hnsw_dir").tempdir().unwrap();
        let quantized_vectors = if quantize {
            let config = QuantizationConfig::Turbo(TurboQuantization {
                turbo: TurboQuantQuantizationConfig {
                    always_ram: Some(true),
                    memory: None,
                    bits: Some(TurboQuantBitSize::Bits4),
                },
            });
            let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
                .vector_storage
                .borrow();
            let quantized = QuantizedVectors::create(
                &storage,
                &config,
                QuantizedVectorsStorageType::Immutable,
                hnsw_dir.path(),
                1,
                &stopped,
            )
            .unwrap();
            Arc::new(AtomicRefCell::new(Some(quantized)))
        } else {
            Default::default()
        };
        let hnsw_config = HnswConfig {
            m: 8,
            ef_construct: 16,
            // In KB: far below the per-value cardinality, so every value makes a block.
            full_scan_threshold: 1,
            max_indexing_threads: 1,
            on_disk: Some(false),
            memory: None,
            payload_m: None,
            inline_storage: None,
        };

        test_hooks::set_force_legacy(legacy);
        test_hooks::take_counts();
        HNSWIndex::build(
            HnswIndexOpenArgs {
                path: hnsw_dir.path(),
                id_tracker: segment.id_tracker.clone(),
                vector_storage: segment.vector_data[DEFAULT_VECTOR_NAME]
                    .vector_storage
                    .clone(),
                quantized_vectors,
                payload_index: segment.payload_index.clone(),
                hnsw_config,
            },
            VectorIndexBuildArgs {
                permit: Arc::new(ResourcePermit::dummy(1)),
                old_indices: &[],
                gpu_device: None,
                rng: &mut StdRng::seed_from_u64(SEED),
                stopped: &stopped,
                hnsw_global_config: &HnswGlobalConfig::default(),
                feature_flags: FeatureFlags::default(),
                inline_vectors: false,
                progress: ProgressTracker::new_for_test(),
            },
        )
        .unwrap();
        test_hooks::set_force_legacy(false);
        let (built, gathered) = test_hooks::take_counts();
        let graph = test_hooks::take_graph().expect("graph was not recorded");
        (graph, built, gathered)
    }

    fn assert_same_graph(legacy: &test_hooks::GraphSnapshot, compact: &test_hooks::GraphSnapshot) {
        assert_eq!(legacy.0.len(), compact.0.len());
        for (point_id, (legacy, compact)) in legacy.0.iter().zip(&compact.0).enumerate() {
            assert_eq!(
                legacy, compact,
                "links of point {point_id} differ between the segment-wide and compact paths",
            );
        }
        assert_eq!(legacy.1, compact.1, "entry points differ");
    }

    /// One build thread, so both builds are deterministic and must match link for link.
    #[rstest]
    #[case::keyword(Field::Keyword)]
    #[case::integer(Field::Integer)]
    #[case::stale_points(Field::KeywordWithStalePoints)]
    fn test_compact_blocks_match_segment_wide(#[case] field: Field) {
        let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();
        let segment = build_segment(dir.path(), field);

        let (legacy, legacy_built, legacy_gathered) = build(&segment, true, false);
        let (compact, compact_built, compact_gathered) = build(&segment, false, false);

        assert!(legacy_built >= 5, "only {legacy_built} blocks were built");
        assert_eq!(legacy_built, compact_built);
        assert_eq!(legacy_gathered, 0);
        // Otherwise the comparison would not cover the gathered scoring path.
        assert_eq!(compact_gathered, compact_built, "gather did not engage");
        assert_same_graph(&legacy, &compact);
    }

    /// Same on TurboQuant 4-bit vectors, where the gathered buffer also serves the
    /// stored-to-stored scoring of the link-selection heuristic.
    #[rstest]
    #[case::keyword(Field::Keyword)]
    #[case::integer(Field::Integer)]
    fn test_compact_blocks_match_segment_wide_quantized(#[case] field: Field) {
        let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();
        let segment = build_segment(dir.path(), field);

        let (legacy, legacy_built, _) = build(&segment, true, true);
        let (compact, compact_built, compact_gathered) = build(&segment, false, true);
        let (raw, _, _) = build(&segment, false, false);

        assert!(legacy_built >= 5, "only {legacy_built} blocks were built");
        assert_eq!(legacy_built, compact_built);
        assert_eq!(compact_gathered, compact_built, "gather did not engage");
        // Quantized scores differ from raw ones, so an equal graph would mean quantization
        // never engaged and the test would cover the wrong path.
        assert_ne!(raw.0, compact.0, "quantization did not engage");
        assert_same_graph(&legacy, &compact);
    }
}
