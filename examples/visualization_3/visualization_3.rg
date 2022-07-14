--
-- visualization_3.rg
--
-- This is a simple proxy application in Regent that requires paraview for visualization.
--


import "regent"
local c = regentlib.c

-------------------------------------------------------------------------------
-- Visualization
-------------------------------------------------------------------------------
local root_dir = arg[0]:match(".*/") or "./"
local mapper = terralib.includec("visualization_3_mapper.h", {"-I", root_dir})
assert(os.getenv('LG_RT_DIR'), "LG_RT_DIR should be set!")
local runtime_dir = os.getenv('LG_RT_DIR') .. "/"
local legion_dir  = runtime_dir .. "legion/"
local mapper_dir  = runtime_dir .. "mappers/"
local realm_dir   = runtime_dir .. "realm/"

local viz_mapper = terralib.includec("visualization_3_mapper.h", {"-I", root_dir})

render = terralib.includec("render.h", {
  "-I", root_dir,
  "-I", runtime_dir,
  "-I", mapper_dir,
  "-I", legion_dir,
  "-I", realm_dir,
})

render.cxx_initialize.replicable = true
render.cxx_terminate.replicable = true
render.cxx_render.replicable = true
render.cxx_reduce.replicable = true
render.cxx_saveImage.replicable = true
render.cxx_saveIndividualImages.replicable = true
render.legion_wait_on_mpi.replicable = true
render.legion_handoff_to_mpi.replicable = true
c.printf.replicable = true

struct Fields {
  TEMP : double
}

local sqrt = regentlib.sqrt(double)

__demand(__cuda)
task perform_time_step(lr : region(ispace(int3d), Fields))
where
  reads writes(lr)
do
  for idx in lr.ispace do
    -- lr[idx].TEMP = c.drand48()
    lr[idx].TEMP = sqrt(idx.x * idx.x + idx.y * idx.y + idx.z * idx.z) / sqrt(128 * 128 * 3)
  end

  -- c.printf("STEP: %d COLOR: %d %d %d Lo: %d %d %d Hi: %d %d %d\n",
  --          step,
  --          color.x, color.y, color.z,
  --          b.lo.x, b.lo.y, b.lo.z,
  --          b.hi.x, b.hi.y, b.hi.z)
end

__demand(__inline)
task initializeVisualization(lr : region(ispace(int3d), Fields),
                             lp : partition(disjoint, lr, ispace(int3d)))
where
  reads(lr.{TEMP})
do
  render.cxx_initialize(__runtime(), __context(),
                        __raw(lr), __raw(lp),
                        __fields(lr.{TEMP}), 1)
end

__demand(__replicable) __forbid(__inner)
task main()
  render.legion_wait_on_mpi();

  var global_grid_size = int3d{120, 120, 120}
  -- var proc_grid_size = int3d{1, 8, 1}
  -- var proc_grid_size = int3d{8, 1, 1}
  -- var proc_grid_size = int3d{2, 2, 2}
  var proc_grid_size = int3d{4, 1, 1}
  -- var proc_grid_size = int3d{1, 1, 1}

  var is_grid = ispace(int3d, global_grid_size)
  var blocking_factor = global_grid_size / proc_grid_size

  var block_size = c.legion_blockify_3d_t{blocking_factor, int3d{0, 0, 0}}

  var ip_rank = c.legion_index_partition_create_blockify_3d(__runtime(), __context(), __raw(is_grid), block_size, -1)
  var raw_ispace = c.legion_index_partition_get_color_space(__runtime(), ip_rank)
  var is_rank = __import_ispace(int3d, raw_ispace)

  var lr_int = region(is_grid, Fields)

  var raw_rank_part = c.legion_logical_partition_create(__runtime(), __raw(lr_int), ip_rank)
  -- var raw_rank_part = c.legion_logical_partition_create(__runtime(), __context(), __raw(lr_int), ip_rank)
  var lp_int_rank = __import_partition(disjoint, lr_int, is_rank, raw_rank_part)

  for color in is_rank do
    var lr = lp_int_rank[color]
    fill(lr.{TEMP}, 1.0)
    c.printf("rank = %i,%i,%i\n", color.x, color.y, color.z)
  end

  initializeVisualization(lr_int, lp_int_rank)
  var camera : render.Camera

  -- camera.from[0] = 146975.3960280538
  -- camera.from[1] = 32651.799693195957
  -- camera.from[2] = 53621.97074250846
  -- camera.at[0] = 7500.0
  -- camera.at[1] = 15508.0
  -- camera.at[2] = 7500.0
  -- camera.up[0] = -0.04362952398351847
  -- camera.up[1] = 0.9723340719573909
  -- camera.up[2] = -0.2294840237309149

  --camera.from[0] = 20.0
  --camera.from[1] = 20.0
  --camera.from[2] = 20.0
  --camera.from[0] = -5.0
  --camera.from[1] = -5.0
  --camera.from[2] = -5.0
  --camera.from[0] = 40.0
  --camera.from[1] = 40.0
  --camera.from[2] = 40.0

  --camera.from[0] = global_grid_size.x * 5
  --camera.from[1] = global_grid_size.y * 5
  --camera.from[2] = global_grid_size.z * 5
  camera.from[0] = global_grid_size.x / 2
  camera.from[1] = global_grid_size.y / 2 * 10
  camera.from[2] = global_grid_size.z / 2

  camera.at[0] = global_grid_size.x / 2
  camera.at[1] = global_grid_size.y / 2
  camera.at[2] = global_grid_size.z / 2
  camera.up[0] = 0.0
  camera.up[1] = 0.0
  camera.up[2] = 1.0

  var args = c.legion_runtime_get_input_args()
  var steps = 10
  for i = 0, args.argc do
    if c.strcmp(args.argv[i], "-t") == 0 then
      steps = c.atoi(args.argv[i+1])
    end
  end

  for step = 0, steps do
    __demand(__index_launch)
    for color in is_rank do
      perform_time_step(lp_int_rank[color])
    end

    if step % 10 == 0 then
      render.cxx_render(__runtime(), __context(), camera, step)
      render.cxx_saveIndividualImages(__runtime(), __context(), "./individual", step)
      render.cxx_reduce(__context(), camera)
      render.cxx_saveImage(__runtime(), __context(), ".", step)
    end
  end

  render.cxx_terminate()

  render.legion_handoff_to_mpi();
end

-- regentlib.saveobj(main, "libviz_3.so", "object", mapper.register_mappers)
main : set_task_id_unsafe(render.TID_TOP_LEVEL_TASK)

local task_whitelist = {}
task_whitelist["main"] = main
regentlib.save_tasks("viz_3.h", "libviz_3.so", nil, nil, "viz_tasks_register", task_whitelist)
