use bevy::mesh::{Indices, Mesh, VertexAttributeValues};

pub use meshopt::SimplifyOptions;

pub trait MeshExt {
    fn simplify(&mut self, params: &SimplifyParams);
    fn optimize_overdraw(&mut self, params: &OverdrawParams);
    fn optimize_vertex_cache(&mut self);
}

#[derive(Debug, Copy, Clone)]
pub enum Target {
    Count(usize),
    Percentage(f32),
}

impl Target {
    pub fn count(&self, current_count: usize) -> usize {
        match self {
            Target::Count(count) => *count,
            Target::Percentage(percentage) => ((percentage * 100.0) as usize / 3 * 3).max(3),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct SimplifyParams<'a> {
    pub max_error: f32,
    pub target_index_count: Target,
    pub options: SimplifyOptions,
    /// Note: Sloppy will ignore all `SimplifyOptions`.
    pub sloppy: bool,
    /// Lock specific vertices in place during simplification.
    pub vertex_locks: Option<&'a [bool]>,
}

#[derive(Debug, Copy, Clone)]
pub enum OptError {
    MissingIndices,
    UnsupportedIndexFormat,
    MissingPositions,
    InvalidIndexCount(usize),
}

fn mesh_indices(mesh: &Mesh) -> Result<&Vec<u32>, OptError> {
    let indices = match mesh.indices() {
        Some(Indices::U32(indices)) => indices,
        Some(_) => return Err(OptError::UnsupportedIndexFormat),
        None => return Err(OptError::MissingIndices),
    };

    if indices.len() % 3 != 0 || indices.len() == 0 {
        return Err(OptError::InvalidIndexCount(indices.len()));
    }

    return Ok(indices);
}

fn mesh_positions(mesh: &Mesh) -> Result<&Vec<[f32; 3]>, OptError> {
    // TODO: support other vertex formats
    let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        return Err(OptError::MissingPositions);
    };

    Ok(positions)
}

impl MeshExt for Mesh {
    fn simplify(&mut self, params: &SimplifyParams) -> Result<f32, OptError> {
        let indices = mesh_indices(self)?;
        let positions = mesh_positions(self)?;

        let target_index_count = params.target_index_count.count(indices.len());

        let mut result_error = 0.0;
        if params.sloppy {
            if let Some(locks) = params.vertex_locks {
                meshopt::simplify_sloppy_with_locks_decoder(
                    indices,
                    &positions,
                    locks,
                    target_index_count,
                    params.max_error,
                    Some(&mut result_error),
                );
            } else {
                meshopt::simplify_sloppy_decoder(
                    indices,
                    positions.as_slice(),
                    target_index_count,
                    params.max_error,
                    Some(&mut result_error),
                );
            }
        } else {
            if let Some(locks) = params.vertex_locks {
                meshopt::simplify_with_locks_decoder(
                    indices,
                    positions.as_slice(),
                    locks,
                    target_index_count,
                    params.max_error,
                    params.options,
                    Some(&mut result_error),
                );
            } else {
                meshopt::simplify_decoder(
                    indices,
                    positions.as_slice(),
                    target_index_count,
                    params.max_error,
                    params.options,
                    Some(&mut result_error),
                );
            }
        }

        // meshopt::optimize_vertex_cache_in_place(
        //     &mut simplified_indices,
        //     combine_data.positions.len() as usize,
        // );
        // meshopt::optimize_overdraw_in_place_decoder(
        //     &mut simplified_indices,
        //     combine_data.positions.as_slice(),
        //     1.1f32,
        // );
        // meshopt::optimize_vertex_fetch_in_place(
        //     &mut simplified_indices,
        //     &mut combine_data.positions,
        // );
        //
        Ok(result_error)
    }

    fn optimize_overdraw(&mut self, params: &OverdrawParams) {}
    fn optimize_vertex_cache(&mut self) {}
}
