package.path = package.path .. ";" .. "/home/jspringer/OpenPV/parameterWrapper/?.lua";
local pv = require "PVModule";
local pvDir = "/home/jspringer/OpenPV";
--local subnets = require "PVSubnets";

local draw = false;

local batchSize     = 25;
local batchWidth    = 25;
local threads       = 1;
local rows          = 1;
local cols          = 1;

local folderName    = "output-noisy-hb";

local nbatch           = batchSize;    -- Batch size of learning
local nxSize           = 1;
local nySize           = 1;
local xPatchSize       = 16;
local yPatchSize       = 16;    -- patch size for V1 to cover all y
local stride           = 2;
local displayPeriod    = 400;   -- Number of timesteps to find sparse approximation
local numEpochs        = 1;     -- Number of times to run through dataset
local numImages        = 75;  -- Total number of images in dataset
local stopTime         = math.ceil((numImages  * numEpochs) / nbatch) * displayPeriod;
local writeStep        = displayPeriod; 
local initialWriteTime = displayPeriod; 

local inputPath        = "/home/jspringer/Workspace/hb_classify/dataset/faces_noisy.txt";
local inputNamePath    = "./dataset/text_nothing.txt";
local inputavg         = "./dataset/avg.txt";
local outputPath       = "./" .. folderName .. "/";
local checkpointPeriod = (displayPeriod * 50); -- How often to write checkpoints

local numBasisVectors  = 128;   --overcompleteness x (stride X) x (Stride Y) * (# color channels) * (2 if rectified) 
local basisVectorFile  = nil;   --nil for initial weights, otherwise, specifies the weights file to load. Change init parameter in MomentumConn
local plasticityFlag   = false;  --Determines if we are learning weights or holding them constant
local momentumTau      = 200;   --The momentum parameter. A single weight update will last for momentumTau timesteps.
local dWMax            = 10; --10;    --The learning rate
local VThresh          = .015;  -- .005; --The threshold, or lambda, of the network
local AMin             = 0;
local AMax             = infinity;
local AShift           = .015;  --This being equal to VThresh is a soft threshold
local VWidth           = 0; 
local timeConstantTau  = 100;   --The integration tau for sparse approximation
local weightInit       = math.sqrt((1/xPatchSize)*(1/yPatchSize)*(1/3));

-- Base table variable to store
local pvParameters = {

   --Layers------------------------------------------------------------
   --------------------------------------------------------------------   
   column = {
      groupType = "HyPerCol";
      startTime                           = 0;
      dt                                  = 1;
      stopTime                            = stopTime;
      progressInterval                    = (displayPeriod * 10);
      writeProgressToErr                  = true;
      verifyWrites                        = false;
      outputPath                          = outputPath;
      printParamsFilename                 = "Multimodal_Tutorial.params";
      randomSeed                          = 1234567890;
      nx                                  = nxSize;
      ny                                  = nySize;
      nbatch                              = nbatch;
      checkpointWrite                     = true;
      checkpointWriteDir                  = outputPath .. "/Checkpoints"; --The checkpoint output directory
      checkpointWriteTriggerMode          = "step";
      checkpointWriteStepInterval         = checkpointPeriod; --How often to checkpoint
      deleteOlderCheckpoints              = false;
      suppressNonplasticCheckpoints       = true;
      errorOnNotANumber                   = false;
   };

   AdaptiveTimeScales = {
      groupType = "AdaptiveTimeScaleProbe";
      targetName                          = "V1EnergyProbe";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "AdaptiveTimeScales.txt";
      triggerLayerName                    = "InputVision";
      triggerOffset                       = 0;
      baseMax                             = .1; --1.0; -- minimum value for the maximum time scale, regardless of tau_eff
      baseMin                             = 0.01; -- default time scale to use after image flips or when something is wacky
      tauFactor                           = 0.1; -- determines fraction of tau_effective to which to set the time step, can be a small percentage as tau_eff can be huge
      growthFactor                        = 0.01; -- percentage increase in the maximum allowed time scale whenever the time scale equals the current maximum
      writeTimeScales                     = true;
   };

   InputVision = {
      groupType = "ImageLayer";
      nxScale                             = 64;
      nyScale                             = 64;
      nf                                  = 3;
      phase                               = 0;
      mirrorBCflag                        = true;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      initializeFromCheckpointFlag        = true;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
      inputPath                           = inputPath;
      offsetAnchor                        = "tl";
      offsetX                             = 0;
      offsetY                             = 0;
      inverseFlag                         = false;
      normalizeLuminanceFlag              = false;
      normalizeStdDev                     = false;
      jitterFlag                          = 0;
      useInputBCflag                      = false;
      padValue                            = 0;
      autoResizeFlag                      = false;
      displayPeriod                       = displayPeriod;
      batchMethod                         = "byImage";
      writeFrameToTimestamp               = true;
      resetToStartOnLoop                  = false;
   };

   InputAverage = {
      groupType = "ImageLayer";
      nxScale                             = 64;
      nyScale                             = 64;
      nf                                  = 3;
      phase                               = 0;
      mirrorBCflag                        = true;
      writeStep                           = -1;
      initialWriteTime                    = initialWriteTime;
      initializeFromCheckpointFlag        = true;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
      inputPath                           = inputavg;
      offsetAnchor                        = "tl";
      offsetX                             = 0;
      offsetY                             = 0;
      inverseFlag                         = false;
      normalizeLuminanceFlag              = false;
      normalizeStdDev                     = false;
      jitterFlag                          = 0;
      useInputBCflag                      = false;
      padValue                            = 0;
      autoResizeFlag                      = false;
      displayPeriod                       = displayPeriod;
      batchMethod                         = "byImage";
      writeFrameToTimestamp               = true;
      resetToStartOnLoop                  = false;
   };

   InputVisionMeanSubtracted = {
      groupType = "HyPerLayer";
      nxScale                             = 64;
      nyScale                             = 64;
      nf                                  = 3;
      phase                               = 1;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = -1;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
   };

  InputVisionMeanSubtractedRescale = {
      groupType = "RescaleLayer";
      nxScale                             = 64;
      nyScale                             = 64;
      nf                                  = 3;
      phase                               = 1;
      targetMean		  	  = 0;
      targetStd				  = 1;
      writeStep                           = -1;
      rescaleMethod			  = "meanstd";
      initialWriteTime                    = initialWriteTime;
      originalLayerName                   = "InputVisionMeanSubtracted";
   };



   InputVisionError = {
      groupType = "HyPerLayer";
      nxScale                             = 64;
      nyScale                             = 64;
      nf                                  = 3;
      phase                               = 3;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
   };

   V1 = {
      groupType = "HyPerLCALayer";
      nxScale                             = 32;  -- 1/2 
      nyScale                             = 32; -- make V1 a size of 1 in the y
      nf                                  = numBasisVectors;
      phase                               = 2;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ConstantV";
      valueV                              = VThresh;
      --InitVType                           = "InitVFromFile";
      --Vfilename                           = "V1_V.pvp";
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = true;
      writeSparseValues                   = true;
      updateGpu                           = true;
      dataType                            = nil;
      VThresh                             = VThresh;
      AMin                                = AMin;
      AMax                                = AMax;
      AShift                              = AShift;
      VWidth                              = VWidth;
      timeConstantTau                     = timeConstantTau;
      selfInteract                        = true;
      adaptiveTimeScaleProbe              = "AdaptiveTimeScales";
   };
   CloneV1 = {
      groupType = "CloneVLayer";
      nxScale                             = 32;
      nyScale                             = 32;
      nf                                  = numBasisVectors;
      phase                               = 2;
      writeStep                           = -1;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
      triggerLayerName                    = NULL;
      originalLayerName                   = "V1";
   };


   V1Error = {
      groupType = "HyPerLayer";
      nxScale                             = 32;
      nyScale                             = 32;
      nf                                  = numBasisVectors;
      phase                               = 1;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = -1;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
 
   };


   InputVisionRecon = {
      groupType = "HyPerLayer";
      nxScale                             = 64;
      nyScale                             = 64;
      nf                                  = 3;
      phase                               = 3;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = -1;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
   };
   InputVisionReconPlusAvg = {
      groupType = "HyPerLayer";
      nxScale                             = 64;
      nyScale                             = 64;
      nf                                  = 3;
      phase                               = 3;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
   };



   V1V2Recon = {
      groupType = "HyPerLayer";
      nxScale                             = 32;
      nyScale                             = 32;
      nf                                  = numBasisVectors;
      phase                               = 3;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = -1;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
   };
  V1V2ThreshRecon = {
      groupType = "HyPerLayer";
      nxScale                             = 32;
      nyScale                             = 32;
      nf                                  = numBasisVectors;
      phase                               = 3;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = -1;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
   };

   V2 = {
      groupType = "HyPerLCALayer";
      nxScale                             = 16;  -- 1/2 
      nyScale                             = 16; -- make V1 a size of 1 in the y
      nf                                  = numBasisVectors;
      phase                               = 2;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ConstantV";
      valueV                              = VThresh;
      --InitVType                           = "InitVFromFile";
      --Vfilename                           = "V1_V.pvp";
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = true;
      writeSparseValues                   = true;
      updateGpu                           = true;
      dataType                            = nil;
      VThresh                             = VThresh;
      AMin                                = AMin;
      AMax                                = AMax;
      AShift                              = AShift;
      VWidth                              = VWidth;
      timeConstantTau                     = timeConstantTau;
      selfInteract                        = true;
      adaptiveTimeScaleProbe              = "AdaptiveTimeScales";
   };
   CloneV2 = {
      groupType = "CloneVLayer";
      nxScale                             = 16;
      nyScale                             = 16;
      nf                                  = numBasisVectors;
      phase                               = 2;
      writeStep                           = -1;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
      triggerLayerName                    = NULL;
      originalLayerName                   = "V2";
   };


   V2Error = {
      groupType = "HyPerLayer";
      nxScale                             = 16;
      nyScale                             = 16;
      nf                                  = numBasisVectors;
      phase                               = 2;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = -1;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
 
   };
   V2P1Recon = {
      groupType = "HyPerLayer";
      nxScale                             = 16;
      nyScale                             = 16;
      nf                                  = numBasisVectors;
      phase                               = 3;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
   };


   InputText = {
      groupType = "ImageLayer";
      nxScale                             = 128;
      nyScale                             = 16;
      nf                                  = 1;
      phase                               = 0;
      mirrorBCflag                        = true;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      initializeFromCheckpointFlag        = true;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
      inputPath                           = inputNamePath;
      offsetAnchor                        = "tl";
      offsetX                             = 0;
      offsetY                             = 0;
      inverseFlag                         = false;
      normalizeLuminanceFlag              = true;
      normalizeStdDev                     = true;
      jitterFlag                          = 0;
      useInputBCflag                      = false;
      padValue                            = 0;
      autoResizeFlag                      = false;
      displayPeriod                       = displayPeriod;
      batchMethod                         = "byImage";
      writeFrameToTimestamp               = true;
      resetToStartOnLoop                  = false;
   };


   InputTextError = {
      groupType = "HyPerLayer";
      nxScale                             = 128;
      nyScale                             = 16;
      nf                                  = 1;
      phase                               = 1;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
   };

   T1 = {
      groupType = "HyPerLCALayer";
      nxScale                             = 64;
      nyScale                             = 1;
      nf                                  = numBasisVectors;
      phase                               = 2;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ConstantV";
      valueV                              = VThresh;
      --InitVType                           = "InitVFromFile";
      --Vfilename                           = "V1_V.pvp";
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = true;
      writeSparseValues                   = true;
      updateGpu                           = true;
      dataType                            = nil;
      VThresh                             = VThresh;
      AMin                                = AMin;
      AMax                                = AMax;
      AShift                              = AShift;
      VWidth                              = VWidth;
      timeConstantTau                     = timeConstantTau;
      selfInteract                        = true;
      adaptiveTimeScaleProbe              = "AdaptiveTimeScales";
   };
   CloneT1 = {
      groupType = "CloneVLayer";
      nxScale                             = 64;
      nyScale                             = 1;
      nf                                  = numBasisVectors;
      phase                               = 2;
      writeStep                           = -1;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
      triggerLayerName                    = NULL;
      originalLayerName                   = "T1";
   };


  T1Error = {
      groupType = "HyPerLayer";
      nxScale                             = 64;
      nyScale                             = 1;
      nf                                  = numBasisVectors;
      phase                               = 1;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = -1;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
   };

   InputTextRecon = {
      groupType = "HyPerLayer";
      nxScale                             = 128;
      nyScale                             = 16;
      nf                                  = 1;
      phase                               = 3;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
   };

   T1P1Recon = {
      groupType = "HyPerLayer";
      nxScale                             = 64;
      nyScale                             = 1;
      nf                                  = numBasisVectors;
      phase                               = 3;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
   };


  ------- multi modal LCA layer ------

  P1 = {
      groupType = "HyPerLCALayer";
      nxScale                             = 1;
      nyScale                             = 1;
      nf                                  = numBasisVectors*4;
      phase                               = 2;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ConstantV";
      valueV                              = VThresh;
      --InitVType                           = "InitVFromFile";
      --Vfilename                           = "V1_V.pvp";
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = true;
      writeSparseValues                   = true;
      updateGpu                           = true;
      dataType                            = nil;
      VThresh                             = VThresh*2;
      AMin                                = AMin;
      AMax                                = AMax;
      AShift                              = AShift*2;
      VWidth                              = VWidth;
      timeConstantTau                     = timeConstantTau;
      selfInteract                        = true;
      adaptiveTimeScaleProbe              = "AdaptiveTimeScales";
   };

   V2applyThresh = {
      groupType = "ANNLayer";
      nxScale                             = 16;
      nyScale                             = 16;
      nf                                  = numBasisVectors;
      phase                               = 5;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ConstantV";
      valueV                              = VThresh;
      --InitVType                           = "InitVFromFile";
      --Vfilename                           = "V1_V.pvp";
      triggerLayerName                    = NULL;
      writeStep                           = -1;
      sparseLayer                         = true;
      writeSparseValues                   = true;
      dataType                            = nil;
      VThresh                             = VThresh;
      AMin                                = AMin;
      AMax                                = AMax;
      AShift                              = AShift;
      VWidth                              = VWidth;
      timeConstantTau                     = timeConstantTau;
      selfInteract                        = true;
   };
   V1V2applyThresh = {
      groupType = "ANNLayer";
      nxScale                             = 32;
      nyScale                             = 32;
      nf                                  = numBasisVectors;
      phase                               = 5;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ConstantV";
      valueV                              = VThresh;
      --InitVType                           = "InitVFromFile";
      --Vfilename                           = "V1_V.pvp";
      triggerLayerName                    = NULL;
      writeStep                           = -1;
      sparseLayer                         = true;
      writeSparseValues                   = true;
      dataType                            = nil;
      VThresh                             = VThresh;
      AMin                                = AMin;
      AMax                                = AMax;
      AShift                              = AShift;
      VWidth                              = VWidth;
      timeConstantTau                     = timeConstantTau;
      selfInteract                        = true;
   };


   T1applyThresh = {
      groupType = "ANNLayer";
      nxScale                             = 64;
      nyScale                             = 1;
      nf                                  = numBasisVectors;
      phase                               = 5;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ConstantV";
      valueV                              = VThresh;
      --InitVType                           = "InitVFromFile";
      --Vfilename                           = "V1_V.pvp";
      triggerLayerName                    = NULL;
      writeStep                           = -1;
      sparseLayer                         = true;
      writeSparseValues                   = true;
      dataType                            = nil;
      VThresh                             = VThresh;
      AMin                                = AMin;
      AMax                                = AMax;
      AShift                              = AShift;
      VWidth                              = VWidth;
      timeConstantTau                     = timeConstantTau;
      selfInteract                        = true;
   };

   P1VisionRecon = {
      groupType = "HyPerLayer";
      nxScale                             = 64;
      nyScale                             = 64;
      nf                                  = 3;
      phase                               = 3;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
   };
   V1V2VisionRecon = {
      groupType = "HyPerLayer";
      nxScale                             = 64;
      nyScale                             = 64;
      nf                                  = 3;
      phase                               = 3;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
   };


   P1TextRecon = {
      groupType = "HyPerLayer";
      nxScale                             = 128;
      nyScale                             = 16;
      nf                                  = 1;
      phase                               = 3;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
   };




--Connections ------------------------------------------------------
--------------------------------------------------------------------
  InputToDiff = {
      groupType = "IdentConn";
      preLayerName                        = "InputVision";
      postLayerName                       = "InputVisionMeanSubtracted";
      channelCode                         = 0;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };
  AvgToDiff = {
      groupType = "IdentConn";
      preLayerName                        = "InputAverage";
      postLayerName                       = "InputVisionMeanSubtracted";
      channelCode                         = 1;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };


   InputToErrorVision = {
      groupType = "RescaleConn";
      preLayerName                        = "InputVisionMeanSubtractedRescale";
      postLayerName                       = "InputVisionError";
      channelCode                         = 0;
      delay                               = {0.000000};
      scale                               = weightInit;
   };

   ErrorToV1 = {
      groupType = "TransposeConn";
      preLayerName                        = "InputVisionError";
      postLayerName                       = "V1";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = true;
      updateGSynFromPostPerspective       = true;
      pvpatchAccumulateType               = "convolve";
      writeStep                           = -1;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      gpuGroupIdx                         = -1;
      originalConnName                    = "V1ToError";
   };

   V1ToError = {
      groupType = "MomentumConn";
      preLayerName                        = "V1";
      postLayerName                       = "InputVisionError";
      channelCode                         = -1;
      delay                               = {0.000000};
      numAxonalArbors                     = 1;
      plasticityFlag                      = plasticityFlag;
      convertRateToSpikeCount             = false;
      receiveGpu                          = false; -- non-sparse -> non-sparse
      sharedWeights                       = true;
      weightInitType                      = "UniformRandomWeight";
      wMinInit                            = -1;
      wMaxInit                            = 1;
      sparseFraction                      = 0.9;
      weightInitType                      = "FileWeight";
      initWeightsFile                     = "./Checkpoint0160000/V1ToError_W.pvp";
      useListOfArborFiles                 = false;
      combineWeightFiles                  = false;
      initializeFromCheckpointFlag        = false;
      triggerLayerName                    = "InputVision";
      triggerOffset                       = 0;
      updateGSynFromPostPerspective       = false; -- Should be false from V1 (sparse layer) to Error (not sparse). Otherwise every input from pre will be calculated (Instead of only active ones)
      pvpatchAccumulateType               = "convolve";
      writeStep                           = -1;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      nxp                                 = 8;
      nyp                                 = 8;
      shrinkPatches                       = false;
      normalizeMethod                     = "normalizeL2";
      strength                            = 1;
      normalizeArborsIndividually         = false;
      normalizeOnInitialize               = true;
      normalizeOnWeightUpdate             = true;
      rMinX                               = 0;
      rMinY                               = 0;
      nonnegativeConstraintFlag           = false;
      normalize_cutoff                    = 0;
      normalizeFromPostPerspective        = false;
      minL2NormTolerated                  = 0;
      dWMax                               = dWMax; 
      useMask                             = false;
      momentumTau                         = momentumTau;   --The momentum parameter. A single weight update will last for momentumTau timesteps.
      momentumMethod                      = "viscosity";
      momentumDecay                       = 0;
   }; 

   V1ToRecon = {
      groupType = "CloneConn";
      preLayerName                        = "V1";
      postLayerName                       = "InputVisionRecon";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = false;
      updateGSynFromPostPerspective       = false;
      pvpatchAccumulateType               = "convolve";
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      originalConnName                    = "V1ToError";
   };
   ReconToOutputPlusAvg = {
      groupType = "IdentConn";
      preLayerName                        = "InputVisionRecon";
      postLayerName                       = "InputVisionReconPlusAvg";
      channelCode                         = 0;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };
   AvgToOutputPlusAvg = {
      groupType = "RescaleConn";
      preLayerName                        = "InputAverage";
      postLayerName                       = "InputVisionReconPlusAvg";
      channelCode                         = -1;
      delay                               = {0.000000};
      scale                               = -1;
   };

   ReconToErrorVision = {
      groupType = "IdentConn";
      preLayerName                        = "InputVisionRecon";
      postLayerName                       = "InputVisionError";
      channelCode                         = 1;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };
   InputToErrorText = {
      groupType = "RescaleConn";
      preLayerName                        = "InputText";
      postLayerName                       = "InputTextError";
      channelCode                         = -1;
      delay                               = {0.000000};
      scale                               = weightInit;
   };

   ErrorToT1 = {
      groupType = "TransposeConn";
      preLayerName                        = "InputTextError";
      postLayerName                       = "T1";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = true;
      updateGSynFromPostPerspective       = true;
      pvpatchAccumulateType               = "convolve";
      writeStep                           = -1;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      gpuGroupIdx                         = -1;
      originalConnName                    = "T1ToError";
   };

   T1ToError = {
      groupType = "MomentumConn";
      preLayerName                        = "T1";
      postLayerName                       = "InputTextError";
      channelCode                         = -1;
      delay                               = {0.000000};
      numAxonalArbors                     = 1;
      plasticityFlag                      = plasticityFlag;
      convertRateToSpikeCount             = false;
      receiveGpu                          = false; -- non-sparse -> non-sparse
      sharedWeights                       = true;
      weightInitType                      = "UniformRandomWeight";
      wMinInit                            = -1;
      wMaxInit                            = 1;
      sparseFraction                      = 0.9;
      weightInitType                      = "FileWeight";
      initWeightsFile                     = "./Checkpoint0160000/T1ToError_W.pvp";
      useListOfArborFiles                 = false;
      combineWeightFiles                  = false;
      initializeFromCheckpointFlag        = false;
      triggerLayerName                    = "InputText";
      triggerOffset                       = 0;
      updateGSynFromPostPerspective       = false; -- Should be false from V1 (sparse layer) to Error (not sparse). Otherwise every input from pre will be calculated (Instead of only active ones)
      pvpatchAccumulateType               = "convolve";
      writeStep                           = -1;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      nxp                                 = 16;
      nyp                                 = 16;
      shrinkPatches                       = false;
      normalizeMethod                     = "normalizeL2";
      strength                            = 1;
      normalizeArborsIndividually         = false;
      normalizeOnInitialize               = true;
      normalizeOnWeightUpdate             = true;
      rMinX                               = 0;
      rMinY                               = 0;
      nonnegativeConstraintFlag           = false;
      normalize_cutoff                    = 0;
      normalizeFromPostPerspective        = false;
      minL2NormTolerated                  = 0;
      dWMax                               = dWMax; 
      useMask                             = false;
      momentumTau                         = momentumTau;   --The momentum parameter. A single weight update will last for momentumTau timesteps.
      momentumMethod                      = "viscosity";
      momentumDecay                       = 0;
   }; 

   T1ToRecon = {
      groupType = "CloneConn";
      preLayerName                        = "T1";
      postLayerName                       = "InputTextRecon";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = false;
      updateGSynFromPostPerspective       = false;
      pvpatchAccumulateType               = "convolve";
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      originalConnName                    = "T1ToError";
   };

   ReconToErrorText = {
      groupType = "IdentConn";
      preLayerName                        = "InputTextRecon";
      postLayerName                       = "InputTextError";
      channelCode                         = 1;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };

   -------Multimdoal Connections ---------
  P1ToT1Error = {
      groupType = "MomentumConn";
      preLayerName                        = "P1";
      postLayerName                       = "T1Error";
      channelCode                         = -1;
      delay                               = {0.000000};
      numAxonalArbors                     = 1;
      plasticityFlag                      = plasticityFlag;
      convertRateToSpikeCount             = false;
      receiveGpu                          = false; -- non-sparse -> non-sparse
      sharedWeights                       = true;
      weightInitType                      = "UniformRandomWeight";
      wMinInit                            = -1;
      wMaxInit                            = 1;
      sparseFraction                      = 0.9;
      weightInitType                      = "FileWeight";
      initWeightsFile                     = "./Checkpoint0160000/P1ToT1Error_W.pvp";
      useListOfArborFiles                 = false;
      combineWeightFiles                  = false;
      initializeFromCheckpointFlag        = false;
      triggerLayerName                    = "InputText";
      triggerOffset                       = 0;
      updateGSynFromPostPerspective       = false; -- Should be false from V1 (sparse layer) to Error (not sparse). Otherwise every input from pre will be calculated (Instead of only active ones)
      pvpatchAccumulateType               = "convolve";
      writeStep                           = -1;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      nxp                                 = 64;
      nyp                                 = 1;
      shrinkPatches                       = false;
      normalizeMethod                     = "normalizeL2";
      strength                            = 1;
      normalizeArborsIndividually         = false;
      normalizeOnInitialize               = true;
      normalizeOnWeightUpdate             = true;
      rMinX                               = 0;
      rMinY                               = 0;
      nonnegativeConstraintFlag           = false;
      normalize_cutoff                    = 0;
      normalizeFromPostPerspective        = false;
      minL2NormTolerated                  = 0;
      dWMax                               = dWMax; 
      useMask                             = false;
      momentumTau                         = momentumTau;   --The momentum parameter. A single weight update will last for momentumTau timesteps.
      momentumMethod                      = "viscosity";
      momentumDecay                       = 0;
   }; 
   T1ErrorToP1 = {
      groupType = "TransposeConn";
      preLayerName                        = "T1Error";
      postLayerName                       = "P1";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = true;
      updateGSynFromPostPerspective       = true;
      pvpatchAccumulateType               = "convolve";
      writeStep                           = -1;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      gpuGroupIdx                         = -1;
      originalConnName                    = "P1ToT1Error";
   };

  V2ToV1Error = {
      groupType = "MomentumConn";
      preLayerName                        = "V2";
      postLayerName                       = "V1Error";
      channelCode                         = -1;
      delay                               = {0.000000};
      numAxonalArbors                     = 1;
      plasticityFlag                      = plasticityFlag;
      convertRateToSpikeCount             = false;
      receiveGpu                          = false; -- non-sparse -> non-sparse
      sharedWeights                       = true;
      weightInitType                      = "UniformRandomWeight";
      wMinInit                            = -1;
      wMaxInit                            = 1;
      sparseFraction                      = 0.9;
      weightInitType                      = "FileWeight";
      initWeightsFile                     = "./Checkpoint0160000/V2ToV1Error_W.pvp";
      useListOfArborFiles                 = false;
      combineWeightFiles                  = false;
      initializeFromCheckpointFlag        = false;
      triggerLayerName                    = "InputVision";
      triggerOffset                       = 0;
      updateGSynFromPostPerspective       = false; -- Should be false from V1 (sparse layer) to Error (not sparse). Otherwise every input from pre will be calculated (Instead of only active ones)
      pvpatchAccumulateType               = "convolve";
      writeStep                           = -1;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      nxp                                 = 8;
      nyp                                 = 8;
      shrinkPatches                       = false;
      normalizeMethod                     = "normalizeL2";
      strength                            = 1;
      normalizeArborsIndividually         = false;
      normalizeOnInitialize               = true;
      normalizeOnWeightUpdate             = true;
      rMinX                               = 0;
      rMinY                               = 0;
      nonnegativeConstraintFlag           = false;
      normalize_cutoff                    = 0;
      normalizeFromPostPerspective        = false;
      minL2NormTolerated                  = 0;
      dWMax                               = dWMax; 
      useMask                             = false;
      momentumTau                         = momentumTau;   --The momentum parameter. A single weight update will last for momentumTau timesteps.
      momentumMethod                      = "viscosity";
      momentumDecay                       = 0;
   }; 
   V1ErrorToV2 = {
      groupType = "TransposeConn";
      preLayerName                        = "V1Error";
      postLayerName                       = "V2";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = true;
      updateGSynFromPostPerspective       = true;
      pvpatchAccumulateType               = "convolve";
      writeStep                           = -1;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      gpuGroupIdx                         = -1;
      originalConnName                    = "V2ToV1Error";
   };
   V1ConeToV1Error = {
      groupType = "IdentConn";
      preLayerName                        = "CloneV1";
      postLayerName                       = "V1Error";
      channelCode                         = 0;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };
  P1ToV2Error = {
      groupType = "MomentumConn";
      preLayerName                        = "P1";
      postLayerName                       = "V2Error";
      channelCode                         = -1;
      delay                               = {0.000000};
      numAxonalArbors                     = 1;
      plasticityFlag                      = plasticityFlag;
      convertRateToSpikeCount             = false;
      receiveGpu                          = false; -- non-sparse -> non-sparse
      sharedWeights                       = true;
      weightInitType                      = "UniformRandomWeight";
      wMinInit                            = -1;
      wMaxInit                            = 1;
      sparseFraction                      = 0.9;
      weightInitType                      = "FileWeight";
      initWeightsFile                     = "./Checkpoint0160000/P1ToV2Error_W.pvp";
      useListOfArborFiles                 = false;
      combineWeightFiles                  = false;
      initializeFromCheckpointFlag        = false;
      triggerLayerName                    = "InputVision";
      triggerOffset                       = 0;
      updateGSynFromPostPerspective       = false; -- Should be false from V1 (sparse layer) to Error (not sparse). Otherwise every input from pre will be calculated (Instead of only active ones)
      pvpatchAccumulateType               = "convolve";
      writeStep                           = -1;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      nxp                                 = 16;
      nyp                                 = 16;
      shrinkPatches                       = false;
      normalizeMethod                     = "normalizeL2";
      strength                            = 1;
      normalizeArborsIndividually         = false;
      normalizeOnInitialize               = true;
      normalizeOnWeightUpdate             = true;
      rMinX                               = 0;
      rMinY                               = 0;
      nonnegativeConstraintFlag           = false;
      normalize_cutoff                    = 0;
      normalizeFromPostPerspective        = false;
      minL2NormTolerated                  = 0;
      dWMax                               = dWMax; 
      useMask                             = false;
      momentumTau                         = momentumTau;   --The momentum parameter. A single weight update will last for momentumTau timesteps.
      momentumMethod                      = "viscosity";
      momentumDecay                       = 0;
   }; 
   V2ErrorToP1 = {
      groupType = "TransposeConn";
      preLayerName                        = "V2Error";
      postLayerName                       = "P1";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = true;
      updateGSynFromPostPerspective       = true;
      pvpatchAccumulateType               = "convolve";
      writeStep                           = -1;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      gpuGroupIdx                         = -1;
      originalConnName                    = "P1ToV2Error";
   };
   V2ConeToV2Error = {
      groupType = "IdentConn";
      preLayerName                        = "CloneV2";
      postLayerName                       = "V2Error";
      channelCode                         = 0;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };

   T1ConeToT1Error = {
      groupType = "IdentConn";
      preLayerName                        = "CloneT1";
      postLayerName                       = "T1Error";
      channelCode                         = 0;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };


   V1ErrorToV1 = {
      groupType = "IdentConn";
      preLayerName                        = "V1Error";
      postLayerName                       = "V1";
      channelCode                         = 1;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };

   V1ReconToV1Error = {
      groupType = "IdentConn";
      preLayerName                        = "V1V2Recon";
      postLayerName                       = "V1Error";
      channelCode                         = 1;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };

   V2ErrorToV2 = {
      groupType = "IdentConn";
      preLayerName                        = "V2Error";
      postLayerName                       = "V2";
      channelCode                         = 1;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };

   V2ReconToV2Error = {
      groupType = "IdentConn";
      preLayerName                        = "V2P1Recon";
      postLayerName                       = "V2Error";
      channelCode                         = 1;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };



   T1ErrorToT1 = {
      groupType = "IdentConn";
      preLayerName                        = "T1Error";
      postLayerName                       = "T1";
      channelCode                         = 1;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };

   T1ReconToT1Error = {
      groupType = "IdentConn";
      preLayerName                        = "T1P1Recon";
      postLayerName                       = "T1Error";
      channelCode                         = 1;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };

   V2ToV1Recon = {
      groupType = "CloneConn";
      preLayerName                        = "V2";
      postLayerName                       = "V1V2Recon";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = false;
      updateGSynFromPostPerspective       = false;
      pvpatchAccumulateType               = "convolve";
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      originalConnName                    = "V2ToV1Error";
   };
   P1ToV2Recon = {
      groupType = "CloneConn";
      preLayerName                        = "P1";
      postLayerName                       = "V2P1Recon";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = false;
      updateGSynFromPostPerspective       = false;
      pvpatchAccumulateType               = "convolve";
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      originalConnName                    = "P1ToV2Error";
   };
  V2ToV1ThreshRecon = {
      groupType = "CloneConn";
      preLayerName                        = "V2applyThresh";
      postLayerName                       = "V1V2ThreshRecon";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = false;
      updateGSynFromPostPerspective       = false;
      pvpatchAccumulateType               = "convolve";
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      originalConnName                    = "V2ToV1Error";
   };



   P1ToT1Recon = {
      groupType = "CloneConn";
      preLayerName                        = "P1";
      postLayerName                       = "T1P1Recon";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = false;
      updateGSynFromPostPerspective       = false;
      pvpatchAccumulateType               = "convolve";
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      originalConnName                    = "P1ToT1Error";
   };
   T1P1ReconToThresh = {
      groupType = "IdentConn";
      preLayerName                        = "T1P1Recon";
      postLayerName                       = "T1applyThresh";
      channelCode                         = 0;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };
   V2P1ReconToThresh = {
      groupType = "IdentConn";
      preLayerName                        = "V2P1Recon";
      postLayerName                       = "V2applyThresh";
      channelCode                         = 0;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };


   P1VisionReconConn = {
      groupType = "CloneConn";
      preLayerName                        = "V1V2ThreshRecon";
      postLayerName                       = "P1VisionRecon";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = false;
      updateGSynFromPostPerspective       = false;
      pvpatchAccumulateType               = "convolve";
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      originalConnName                    = "V1ToError";
   };
   V1V2VisionReconConn = {
      groupType = "CloneConn";
      preLayerName                        = "V1V2applyThresh";
      postLayerName                       = "V1V2VisionRecon";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = false;
      updateGSynFromPostPerspective       = false;
      pvpatchAccumulateType               = "convolve";
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      originalConnName                    = "V1ToError";
   };
   V1V2ReconToThreshL2 = {
      groupType = "IdentConn";
      preLayerName                        = "V1V2Recon";
      postLayerName                       = "V1V2applyThresh";
      channelCode                         = 0;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };



   T1VisionReconConn = {
      groupType = "CloneConn";
      preLayerName                        = "T1applyThresh";
      postLayerName                       = "P1TextRecon";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = false;
      updateGSynFromPostPerspective       = false;
      pvpatchAccumulateType               = "convolve";
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      originalConnName                    = "T1ToError";
   };

   ----- Visualization of Activity Triggered Average ------
   -------------------------------------------------------
 P1ToInputVision = {
      groupType = "HyPerConn";
      preLayerName                        = "P1";
      postLayerName                       = "InputVision";
      channelCode                         = -1;
      delay                               = {0.000000};
      numAxonalArbors                     = 1;
      plasticityFlag                      = plasticityFlag;
      convertRateToSpikeCount             = false;
      receiveGpu                          = false; -- non-sparse -> non-sparse
      sharedWeights                       = true;
      weightInitType                      = "UniformWeight";
      weightInit                          = 0;
      --weightInitType                      = "FileWeight";
      --initWeightsFile                     = "../outputCheck/Checkpoints/Checkpoint040000/P1ToV1Error_W.pvp";
      useListOfArborFiles                 = false;
      combineWeightFiles                  = false;
      initializeFromCheckpointFlag        = false;
      triggerLayerName                    = "InputVision";
      triggerOffset                       = 0;
      updateGSynFromPostPerspective       = false; -- Should be false from V1 (sparse layer) to Error (not sparse). Otherwise every input from pre will be calculated (Instead of only active ones)
      pvpatchAccumulateType               = "convolve";
      writeStep                           = writeStep*3;
      initialWriteTime                    = initialWriteTime;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      nxp                                 = 64;
      nyp                                 = 64;
      nfp                                 = 3;
      shrinkPatches                       = false;
      normalizeMethod                     = "none";
      rMinX                               = 0;
      rMinY                               = 0;
      nonnegativeConstraintFlag           = false;
      normalize_cutoff                    = 0;
      normalizeFromPostPerspective        = false;
      minL2NormTolerated                  = 0;
      dWMax                               = 1; 
      useMask                             = false;
   }; 
 P1ToInputText = {
      groupType = "HyPerConn";
      preLayerName                        = "P1";
      postLayerName                       = "InputText";
      channelCode                         = -1;
      delay                               = {0.000000};
      numAxonalArbors                     = 1;
      plasticityFlag                      = plasticityFlag;
      convertRateToSpikeCount             = false;
      receiveGpu                          = false; -- non-sparse -> non-sparse
      sharedWeights                       = true;
      weightInitType                      = "UniformWeight";
      weightInit                          = 0;
      --weightInitType                      = "FileWeight";
      --initWeightsFile                     = "../outputCheck/Checkpoints/Checkpoint040000/P1ToV1Error_W.pvp";
      useListOfArborFiles                 = false;
      combineWeightFiles                  = false;
      initializeFromCheckpointFlag        = false;
      triggerLayerName                    = "InputVision";
      triggerOffset                       = 0;
      updateGSynFromPostPerspective       = false; -- Should be false from V1 (sparse layer) to Error (not sparse). Otherwise every input from pre will be calculated (Instead of only active ones)
      pvpatchAccumulateType               = "convolve";
      writeStep                           = writeStep*3;
      initialWriteTime                    = initialWriteTime;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      nxp                                 = 128;
      nyp                                 = 16;
      nfp                                 = 1;
      shrinkPatches                       = false;
      normalizeMethod                     = "none";
      rMinX                               = 0;
      rMinY                               = 0;
      nonnegativeConstraintFlag           = false;
      normalize_cutoff                    = 0;
      normalizeFromPostPerspective        = false;
      minL2NormTolerated                  = 0;
      dWMax                               = 1; 
      useMask                             = false;
   }; 


   --Probes------------------------------------------------------------
   --------------------------------------------------------------------

   V1EnergyProbe = {
      groupType = "ColumnEnergyProbe";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "V1EnergyProbe.txt";
      triggerLayerName                    = nil;
      energyProbe                         = nil;
   };
   V2EnergyProbe = {
      groupType = "ColumnEnergyProbe";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "V2EnergyProbe.txt";
      triggerLayerName                    = nil;
      energyProbe                         = "V1EnergyProbe";
   };

   T1EnergyProbe = {
      groupType = "ColumnEnergyProbe";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "T1EnergyProbe.txt";
      triggerLayerName                    = nil;
      energyProbe                         = "V1EnergyProbe";
   };
   P1EnergyProbe = {
      groupType = "ColumnEnergyProbe";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "P1EnergyProbe.txt";
      triggerLayerName                    = nil;
      energyProbe                         = "V1EnergyProbe";
   };

   InputVisionErrorL2NormEnergyProbe = {
      groupType = "L2NormProbe";
      targetLayer                         = "InputVisionError";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "InputVisionErrorL2NormEnergyProbe.txt";
      energyProbe                         = "V1EnergyProbe";
      coefficient                         = 0.5;
      maskLayerName                       = nil;
      exponent                            = 2;
   };
   InputTextErrorL2NormEnergyProbe = {
      groupType = "L2NormProbe";
      targetLayer                         = "InputTextError";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "InputTextErrorL2NormEnergyProbe.txt";
      energyProbe                         = "T1EnergyProbe";
      coefficient                         = 0.5;
      maskLayerName                       = nil;
      exponent                            = 2;
   };
  InputV1ErrorL2NormEnergyProbe = {
      groupType = "L2NormProbe";
      targetLayer                         = "V1Error";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "V1ErrorL2NormEnergyProbe.txt";
      energyProbe                         = "V2EnergyProbe";
      coefficient                         = 0.5;
      maskLayerName                       = nil;
      exponent                            = 2;
   };
  InputV2ErrorL2NormEnergyProbe = {
      groupType = "L2NormProbe";
      targetLayer                         = "V2Error";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "V2ErrorL2NormEnergyProbe.txt";
      energyProbe                         = "P1EnergyProbe";
      coefficient                         = 0.5;
      maskLayerName                       = nil;
      exponent                            = 2;
   };

  InputT1ErrorL2NormEnergyProbe = {
      groupType = "L2NormProbe";
      targetLayer                         = "T1Error";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "T1ErrorL2NormEnergyProbe.txt";
      energyProbe                         = "P1EnergyProbe";
      coefficient                         = 0.5;
      maskLayerName                       = nil;
      exponent                            = 2;
   };


   V1L1NormEnergyProbe = {
      groupType = "L1NormProbe";
      targetLayer                         = "V1";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "V1L1NormEnergyProbe.txt";
      energyProbe                         = "V1EnergyProbe";
      coefficient                         = 0.025;
      maskLayerName                       = nil;
   };
   V2L1NormEnergyProbe = {
      groupType = "L1NormProbe";
      targetLayer                         = "V2";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "V2L1NormEnergyProbe.txt";
      energyProbe                         = "V2EnergyProbe";
      coefficient                         = 0.025;
      maskLayerName                       = nil;
   };

   T1L1NormEnergyProbe = {
      groupType = "L1NormProbe";
      targetLayer                         = "T1";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "T1L1NormEnergyProbe.txt";
      energyProbe                         = "T1EnergyProbe";
      coefficient                         = 0.025;
      maskLayerName                       = nil;
   };
   P1L1NormEnergyProbe = {
      groupType = "L1NormProbe";
      targetLayer                         = "P1";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "P1L1NormEnergyProbe.txt";
      energyProbe                         = "P1EnergyProbe";
      coefficient                         = 0.025;
      maskLayerName                       = nil;
   };




} --End of pvParameters

-- Build and run the params file --

os.execute("mkdir -p " .. folderName);
local file = io.open(folderName .. "/temp.params", "w");
io.output(file);
pv.printConsole(pvParameters);
io.close(file);
if draw then
  -- The & makes it run without blocking execution
  os.execute(pvDir .. "/python/draw -p -a " .. folderName .. "/temp.params &");
end
os.execute("mpiexec -np " .. batchWidth * rows * cols .. " " .. pvDir .. "/build/tests/BasicSystemTest/Release/BasicSystemTest -p " .. folderName .. "/temp.params -t " .. threads .. " -batchwidth " .. batchWidth .. " -rows " .. rows .. " -columns " .. cols);
-- Make sure we close the draw tool
if draw then
   os.execute("killall draw -KILL");
end
