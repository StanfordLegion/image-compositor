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

struct Model {
  xMin : float,
  yMin : float,
  zMin : float,
  xMax : float,
  yMax : float,
  zMax : float,
  numNodes : int[3]
}

struct Camera {
  from : float[3],
  at : float[3],
  up : float[3]
}

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




task configureCamera(model : Model, angle : float) 
-- compute from, at, up
  var camera : Camera
  return camera
end



task renderScene(
  camera : Camera,
  model : Model,
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
    numPFields)
end



task compositeImages() 
  render.cxx_reduce(__runtime(), __context(), ".")
end



task saveImage() 

end



task main() 

  -- geometry model

  var model : Model
  model.xMin = 0
  model.yMin = 0
  model.zMin = 0
  model.xMax = 1
  model.yMax = 1
  model.zMax = 1
  model.numNodes[0] = 2
  model.numNodes[1] = 2
  model.numNodes[2] = 2
  var totalNodes : int = 8

  -- logical region and partition

  var r = region(ispace(int3d, {2,2,2}, {0,0,0}), float)
  var colors = ispace(int3d, {1, 1, totalNodes}, {0, 0, 0})
  var p = partition(equal, r, colors)

  var viz = renderInitialize(r, colors, p)

  for angle = 0, 360 do
    var camera : Camera
    camera = configureCamera(model, angle)
    renderScene(camera, model, r, colors, p, viz)
    compositeImages()
    saveImage()
  end
end

regentlib.saveobj(main, "visualization_2.so", "object", MAPPER.register_mappers)

