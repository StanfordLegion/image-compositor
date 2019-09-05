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



task renderInitialize(
  r : region(ispace(int3d), float),
  colors : ispace(int3d),
  p : partition(disjoint, r, colors)
) 
  var result = render.cxx_initialize(__runtime(), __context(), __raw(p))
  return result
end




task configureCamera(angle : float)
-- compute from, at, up
  var camera : render.Camera
  return camera
end



task renderScene(
  camera : render.Camera,
  r : region(ispace(int3d), float),
  colors : ispace(int3d),
  p : partition(disjoint, r, colors),
  viz : render.RegionPartition) 

  var indexSpace = __import_ispace(int3d, viz.indexSpace)
  var imageX = __import_region(indexSpace, Image_columns, viz.imageX, viz.imageFields)

  var numImageFields : int = 6
  var numPFields : int = 1


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
end



task compositeImages() 
  render.cxx_reduce(__runtime(), __context())
end



task saveImage() 
  render.cxx_saveImage(__runtime(), __context(), ".")
end



task main() 

  -- logical region and partition

  var r = region(ispace(int3d, {2, 2, 2}, {0, 0, 0}), float)
  var colors = ispace(int3d, {1, 1, 8}, {0, 0, 0})
  var p = partition(equal, r, colors)

  var viz = renderInitialize(r, colors, p)

  for angle = 0, 360 do
    var camera = configureCamera(angle)
    renderScene(camera, r, colors, p, viz)
    compositeImages()
    saveImage()
  end
end

regentlib.saveobj(main, "visualization_2.so", "object", MAPPER.register_mappers)

