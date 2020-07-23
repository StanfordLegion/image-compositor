
#--------------------------------------------------------------

# Global timestep output options
timeStepToStartOutputAt=0
forceOutputAtFirstCall=False

# Global screenshot output options
imageFileNamePadding=0
rescale_lookuptable=False

# Whether or not to request specific arrays from the adaptor.
requestSpecificArrays=False

# a root directory under which all Catalyst output goes
rootDirectory=''

# makes a cinema D index table
make_cinema_table=False

#--------------------------------------------------------------
# Code generated from cpstate.py to create the CoProcessor.
# paraview version 5.8.0
#--------------------------------------------------------------

from paraview.simple import *
from paraview import coprocessing
import vtk
from vtk.util import numpy_support
import numpy as np

# ----------------------- CoProcessor definition -----------------------

def CreateCoProcessor():
  def _CreatePipeline(coprocessor, datadescription):
    class Pipeline:
      # state file generated using paraview version 5.8.0

      # ----------------------------------------------------------------
      # setup views used in the visualization
      # ----------------------------------------------------------------

      # trace generated using paraview version 5.8.0
      #
      # To ensure correct image size when batch processing, please search 
      # for and uncomment the line `# renderView*.ViewSize = [*,*]`

      #### disable automatic camera reset on 'Show'
      paraview.simple._DisableFirstRenderCameraReset()

      # get the material library
      materialLibrary1 = GetMaterialLibrary()

      # Create a new 'Render View'
      renderView1 = CreateView('RenderView')
      renderView1.ViewSize = [3222, 1180]
      renderView1.AxesGrid = 'GridAxes3DActor'
      renderView1.CenterOfRotation = [7500.0, 15508.0, 7500.0]
      renderView1.StereoType = 'Crystal Eyes'
      renderView1.CameraPosition = [146975.3960280538, 32651.799693195957, 53621.97074250846]
      renderView1.CameraFocalPoint = [7500.0, 15508.0, 7500.0]
      renderView1.CameraViewUp = [-0.04362952398351847, 0.9723340719573909, -0.2294840237309149]
      renderView1.CameraFocalDisk = 1.0
      renderView1.CameraParallelScale = 39308.734740641485
      renderView1.BackEnd = 'OSPRay raycaster'
      renderView1.OSPRayMaterialLibrary = materialLibrary1

      # register the view with coprocessor
      # and provide it with information such as the filename to use,
      # how frequently to write the images, etc.
      coprocessor.RegisterView(renderView1,
          filename='RenderView1_%t.png', freq=1, fittoscreen=0, magnification=1, width=2430, height=1180, cinema={}, compression=5)
      renderView1.ViewTime = datadescription.GetTime()

      SetActiveView(None)

      # ----------------------------------------------------------------
      # setup view layouts
      # ----------------------------------------------------------------

      # create new layout object 'Layout #1'
      layout1 = CreateLayout(name='Layout #1')
      layout1.AssignView(0, renderView1)

      # ----------------------------------------------------------------
      # restore active view
      SetActiveView(renderView1)
      # ----------------------------------------------------------------

      # ----------------------------------------------------------------
      # setup the data processing pipelines
      # ----------------------------------------------------------------

      # create a new 'XML Partitioned Image Data Reader'
      # create a producer from a simulation input
      tEMP = coprocessor.CreateProducer(datadescription, 'TEMP')

      # create a new 'XML Partitioned Image Data Reader'
      # create a producer from a simulation input
      tEMP_1 = coprocessor.CreateProducer(datadescription, 'TEMP')

      # create a new 'Append Datasets'
      appendDatasets1 = AppendDatasets(Input=[tEMP_1, tEMP])

      # ----------------------------------------------------------------
      # setup the visualization in view 'renderView1'
      # ----------------------------------------------------------------

      # show data from appendDatasets1
      appendDatasets1Display = Show(appendDatasets1, renderView1, 'UnstructuredGridRepresentation')

      # get color transfer function/color map for 'TEMP'
      tEMPLUT = GetColorTransferFunction('TEMP')
      tEMPLUT.AutomaticRescaleRangeMode = 'Grow and update every timestep'
      tEMPLUT.EnableOpacityMapping = 1
      tEMPLUT.RGBPoints = [11.0, 0.231373, 0.298039, 0.752941, 461.0625, 0.865003, 0.865003, 0.865003, 911.125, 0.705882, 0.0156863, 0.14902]
      tEMPLUT.ScalarRangeInitialized = 1.0

      # get opacity transfer function/opacity map for 'TEMP'
      tEMPPWF = GetOpacityTransferFunction('TEMP')
      tEMPPWF.Points = [11.0, 0.0, 0.5, 0.0, 911.125, 1.0, 0.5, 0.0]
      tEMPPWF.ScalarRangeInitialized = 1

      # trace defaults for the display properties.
      appendDatasets1Display.Representation = 'Volume'
      appendDatasets1Display.ColorArrayName = ['POINTS', 'TEMP']
      appendDatasets1Display.LookupTable = tEMPLUT
      appendDatasets1Display.OSPRayScaleArray = 'TEMP'
      appendDatasets1Display.OSPRayScaleFunction = 'PiecewiseFunction'
      appendDatasets1Display.SelectOrientationVectors = 'None'
      appendDatasets1Display.ScaleFactor = 3101.6000000000004
      appendDatasets1Display.SelectScaleArray = 'None'
      appendDatasets1Display.GlyphType = 'Arrow'
      appendDatasets1Display.GlyphTableIndexArray = 'None'
      appendDatasets1Display.GaussianRadius = 155.08
      appendDatasets1Display.SetScaleArray = ['POINTS', 'TEMP']
      appendDatasets1Display.ScaleTransferFunction = 'PiecewiseFunction'
      appendDatasets1Display.OpacityArray = ['POINTS', 'TEMP']
      appendDatasets1Display.OpacityTransferFunction = 'PiecewiseFunction'
      appendDatasets1Display.DataAxesGrid = 'GridAxesRepresentation'
      appendDatasets1Display.PolarAxes = 'PolarAxesRepresentation'
      appendDatasets1Display.ScalarOpacityFunction = tEMPPWF
      appendDatasets1Display.ScalarOpacityUnitDistance = 1988.298415800109

      # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
      appendDatasets1Display.ScaleTransferFunction.Points = [911.0, 0.0, 0.5, 0.0, 911.125, 1.0, 0.5, 0.0]

      # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
      appendDatasets1Display.OpacityTransferFunction.Points = [911.0, 0.0, 0.5, 0.0, 911.125, 1.0, 0.5, 0.0]

      # setup the color legend parameters for each legend in this view

      # get color legend/bar for tEMPLUT in view renderView1
      tEMPLUTColorBar = GetScalarBar(tEMPLUT, renderView1)
      tEMPLUTColorBar.Title = 'TEMP'
      tEMPLUTColorBar.ComponentTitle = ''

      # set color bar visibility
      tEMPLUTColorBar.Visibility = 1

      # show color legend
      appendDatasets1Display.SetScalarBarVisibility(renderView1, True)

      # ----------------------------------------------------------------
      # setup color maps and opacity mapes used in the visualization
      # note: the Get..() functions create a new object, if needed
      # ----------------------------------------------------------------

      # ----------------------------------------------------------------
      # finally, restore active source
      SetActiveSource(appendDatasets1)
      # ----------------------------------------------------------------

    return Pipeline()

  class CoProcessor(coprocessing.CoProcessor):
    def CreatePipeline(self, datadescription):
      self.Pipeline = _CreatePipeline(self, datadescription)

  coprocessor = CoProcessor()
  # these are the frequencies at which the coprocessor updates.
  freqs = {'TEMP': [1]}
  coprocessor.SetUpdateFrequencies(freqs)
  if requestSpecificArrays:
    arrays = [['TEMP', 0]]
    coprocessor.SetRequestedArrays('TEMP', arrays)
  coprocessor.SetInitialOutputOptions(timeStepToStartOutputAt,forceOutputAtFirstCall)

  if rootDirectory:
      coprocessor.SetRootDirectory(rootDirectory)

  if make_cinema_table:
      coprocessor.EnableCinemaDTable()

  return coprocessor


#--------------------------------------------------------------
# Global variable that will hold the pipeline for each timestep
# Creating the CoProcessor object, doesn't actually create the ParaView pipeline.
# It will be automatically setup when coprocessor.UpdateProducers() is called the
# first time.
coprocessor = CreateCoProcessor()

#--------------------------------------------------------------
# Enable Live-Visualizaton with ParaView and the update frequency
coprocessor.EnableLiveVisualization(False, 1)

# ---------------------- Data Selection method ----------------------

def RequestDataDescription(datadescription):
    "Callback to populate the request for current timestep"
    global coprocessor

    # setup requests for all inputs based on the requirements of the
    # pipeline.
    coprocessor.LoadRequestedData(datadescription)

# ------------------------ Processing method ------------------------

def DoCoProcessing(datadescription):
    "Callback to do co-processing for current timestep"
    global coprocessor

    # Update the coprocessor by providing it the newly generated simulation data.
    # If the pipeline hasn't been setup yet, this will setup the pipeline.
    coprocessor.UpdateProducers(datadescription)

    # Write output data, if appropriate.
    coprocessor.WriteData(datadescription);

    # Write image capture (Last arg: rescale lookup table), if appropriate.
    coprocessor.WriteImages(datadescription, rescale_lookuptable=rescale_lookuptable,
        image_quality=0, padding_amount=imageFileNamePadding)

    # Live Visualization, if enabled.
    coprocessor.DoLiveVisualization(datadescription, "localhost", 22222)

    view = GetActiveView()
    ss = vtk.vtkWindowToImageFilter()
    ss.SetInputBufferTypeToZBuffer()
    ss.SetInput(view.SMProxy.GetRenderWindow())
    ss.Update()
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetInputConnection(ss.GetOutputPort())
    writer.SetFileName("z_buffer_%d.vti" % (view.ViewTime * 10))
    writer.Write()
