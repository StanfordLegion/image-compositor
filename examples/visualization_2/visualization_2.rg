--
-- visualization_2.rg
--
-- This is a simple proxy application in Regent that requires blending for visualization.
--


import "regent"
c = regentlib.c


-------------------------------------------------------------------------------
-- Visualization
-------------------------------------------------------------------------------

local root_dir = arg[0]:match(".*/") or "./"
assert(os.getenv('LG_RT_DIR'), "LG_RT_DIR should be set!")
local runtime_dir = os.getenv('LG_RT_DIR') .. "/"
local legion_dir = runtime_dir .. "legion/"
local mapper_dir = runtime_dir .. "mappers/"
local realm_dir = runtime_dir .. "realm/"

render = terralib.includec("render.h",
{"-I", root_dir,
"-I", runtime_dir,
"-I", mapper_dir,
"-I", legion_dir,
"-I", realm_dir,
})

struct Image_columns {
  R : float,
  G : float,
  B : float,
  A : float,
  Z : float,
  U : float
}




terra configureCamera(angle : float)
  var camera : render.Camera
  if angle < -180 then angle = angle + 360 end
  if angle > 180 then angle = angle - 360 end
  camera.up[0] = 0
  camera.up[1] = 1
  camera.up[2] = 0
  camera.from[0] = c.cos(angle) * 6
  camera.from[1] = -1.0 + 4 * c.sin(angle);
  camera.from[2] = c.sin(angle) * 6
  camera.at[0] = 1
  camera.at[1] = 1
  camera.at[2] = 1
  return camera
end



__forbid(__inner)
task renderLoop(
  r : region(ispace(int3d), float),
  colors : ispace(int3d),
  p : partition(disjoint, r, colors)
)
where reads(r)
do
  render.cxx_initialize(__runtime(), __context(), __raw(r), __raw(p),
    __fields(r), 1)

  var stepsPerAngle = 100
  for loop = 0, 180 * stepsPerAngle do
    var angle : float = loop * (1.0 / stepsPerAngle)
    var camera = configureCamera(angle)
    render.cxx_render(__runtime(), __context(), camera)
    var direction : float[3]
    for i = 0, 3 do
      direction[i] = camera.at[i] - camera.from[i]
    end
    render.cxx_reduce(__context(), direction)
    render.cxx_saveImage(__runtime(), __context(), ".")
  end
end



task main()

  -- logical region and partition
  var r = region(ispace(int3d, {2, 2, 2}, {0, 0, 0}), float)
  var colors = ispace(int3d, {2, 2, 2}, {0, 0, 0})
  var p = partition(equal, r, colors)
  fill(r, 0.0)
  renderLoop(r, colors, p)
  render.cxx_terminate()
end

regentlib.saveobj(main, "visualization_2.so", "object", render.cxx_preinitialize)
