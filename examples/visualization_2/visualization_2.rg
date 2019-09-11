--
-- visualization_2.rg
--
-- This is a simple proxy application in Regent that requires blending for visualization.
--


import "regent"
c = regentlib.c
local MAPPER = terralib.includec("visualization_2_mapper.h")



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
  camera.up[0] = 0
  camera.up[1] = 1
  camera.up[2] = 0
  camera.from[0] = c.cos(angle) * 4
  camera.from[1] = 1.5
  camera.from[2] = c.sin(angle) * 4
  camera.at[0] = 1
  camera.at[1] = 1
  camera.at[2] = 1
  return camera
end





task main() 

  -- logical region and partition

  var r = region(ispace(int3d, {2, 2, 2}, {0, 0, 0}), float)
  var colors = ispace(int3d, {2, 2, 2}, {0, 0, 0})
  var p = partition(equal, r, colors)
  fill(r, 0.0)

  var viz = render.cxx_initialize(__runtime(), __context(), __raw(p))
  var indexSpace = __import_ispace(int3d, viz.indexSpace)
  var imageX = __import_region(indexSpace, Image_columns, viz.imageX, viz.imageFields)
  var numImageFields : int = 6
  var numPFields : int = 1

  for angle = 0, 1 do -- 360 do
    var camera = configureCamera(angle)
    render.cxx_render(__runtime(),
      __context(),
      __physical(imageX),
      viz.imageFields,
      numImageFields,
      __raw(r),
      __raw(p),
      __fields(r),
      numPFields,
      camera)
    if render.DEBUG_INDIVIDUAL_IMAGES == 1 then
      render.cxx_saveIndividualImages(__runtime(), __context(), ".")
    else
      render.cxx_reduce(__context())
      render.cxx_saveImage(__runtime(), __context(), ".")
    end
  end

  render.cxx_terminate()
end

regentlib.saveobj(main, "visualization_2.so", "object", MAPPER.register_mappers)

