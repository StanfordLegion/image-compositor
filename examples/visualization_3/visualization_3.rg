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
local legion_dir = runtime_dir .. "legion/"
local mapper_dir = runtime_dir .. "mappers/"
local realm_dir = runtime_dir .. "realm/"

local viz_mapper = terralib.includec("visualization_3_mapper.h", {"-I", root_dir})

render = terralib.includec("render.h",
{"-I", root_dir,
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

-- terra configureCamera(angle : float)
--   var camera : render.Camera
--   if angle < -180 then angle = angle + 360 end
--   if angle > 180 then angle = angle - 360 end
--   camera.up[0] = 0
--   camera.up[1] = 1
--   camera.up[2] = 0
--   camera.from[0] = c.cos(angle) * 6
--   --camera.from[1] = -1.0 + 4 * c.sin(angle);
--   camera.from[1] = 1.5
--   camera.from[2] = c.sin(angle) * 6
--   camera.at[0] = 1
--   camera.at[1] = 1
--   camera.at[2] = 1
--   return camera
-- end


-- __forbid(__inner)
-- task renderLoop(
--   r : region(ispace(int3d), Fields),
--   colors : ispace(int3d),
--   p : partition(disjoint, r, colors)
-- )
-- where reads(r)
-- do
--   render.cxx_initialize(__runtime(), __context(), __raw(r), __raw(p),
--                         __fields(r.{TEMP}), 1)

--   var stepsPerAngle = 100
--   var angles = 180

--   for loop = 0, angles * stepsPerAngle do
--     var angle : float = loop * (1.0 / stepsPerAngle)
--     var camera = configureCamera(angle)
--     render.cxx_render(__runtime(), __context(), camera)
--     var direction : float[3]
--     for i = 0, 3 do
--       direction[i] = camera.at[i] - camera.from[i]
--     end
--     --render.cxx_saveIndividualImages(__runtime(), __context(), ".")
--     render.cxx_reduce(__context(), direction)
--     render.cxx_saveImage(__runtime(), __context(), ".")
--   end
-- end

task perform_time_step(lr : region(ispace(int3d), Fields))
where
  reads writes(lr)
do
  for idx in lr.ispace do
    -- lr[idx].TEMP = c.drand48()
    lr[idx].TEMP += 10
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

  var global_grid_size = int3d{16, 32, 16}
  var proc_grid_size = int3d{1, 2, 1}

  var is_grid = ispace(int3d, global_grid_size)
  var blocking_factor = global_grid_size / proc_grid_size

  var block_size = c.legion_blockify_3d_t{blocking_factor, int3d{0, 0, 0}}

  var ip_rank = c.legion_index_partition_create_blockify_3d(__runtime(), __context(),
                                                            __raw(is_grid), block_size, -1)
  var raw_ispace = c.legion_index_partition_get_color_space(__runtime(), ip_rank)
  var is_rank = __import_ispace(int3d, raw_ispace)

  var lr_int = region(is_grid, Fields)

  var raw_rank_part = c.legion_logical_partition_create(__runtime(), __context(), __raw(lr_int), ip_rank)
  var lp_int_rank = __import_partition(disjoint, lr_int, is_rank, raw_rank_part)

  for color in is_rank do
    var lr = lp_int_rank[color]
    fill(lr.{TEMP}, 1.0)
  end

  initializeVisualization(lr_int, lp_int_rank)
  var camera : render.Camera
  camera.from[0] = -42300.32384962992
  camera.from[1] = 12359.491036211044
  camera.from[2] = -60266.278504190705
  camera.at[0] = 21305.058773578516
  camera.at[1] = 6152.909174503882
  camera.at[2] = 26285.36895546706
  camera.up[0] = -0.5336931391138512
  camera.up[1] = 0.7198561483658373
  camera.up[2] = 0.4438228913910425

  var args = c.legion_runtime_get_input_args()
  var steps = 100
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
      render.cxx_saveIndividualImages(__runtime(), __context(), ".")
      render.cxx_reduce(__context(), camera)
      render.cxx_saveImage(__runtime(), __context(), ".")
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
