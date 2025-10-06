import React, { useState, useEffect, useRef, useCallback } from 'react';
import { HandLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';

const HandDetector: React.FC = () => {
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isWebcamActive, setIsWebcamActive] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [detectedHandsCount, setDetectedHandsCount] = useState<number>(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const handCanvasRef1 = useRef<HTMLCanvasElement>(null);
  const handCanvasRef2 = useRef<HTMLCanvasElement>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const animationFrameId = useRef<number | null>(null);
  const drawingUtilsRef = useRef<DrawingUtils | null>(null);
  const handDrawingUtilsRef1 = useRef<DrawingUtils | null>(null);
  const handDrawingUtilsRef2 = useRef<DrawingUtils | null>(null);

  const createHandLandmarker = useCallback(async () => {
    try {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );
      const landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
      });
      handLandmarkerRef.current = landmarker;
      setIsLoading(false);
    } catch (err) {
      console.error("Error creating HandLandmarker:", err);
      setError("Failed to load the hand detection model. Please try refreshing the page.");
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    createHandLandmarker();
  }, [createHandLandmarker]);

  const predictWebcam = useCallback(() => {
    if (!handLandmarkerRef.current || !videoRef.current || !canvasRef.current || !isWebcamActive) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const canvasCtx = canvas.getContext("2d");
    if (!canvasCtx) return;
    
    if (!drawingUtilsRef.current) {
        drawingUtilsRef.current = new DrawingUtils(canvasCtx);
    }

    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const startTimeMs = performance.now();
    const results = handLandmarkerRef.current.detectForVideo(video, startTimeMs);
    
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.landmarks) {
      for (const landmarks of results.landmarks) {
        drawingUtilsRef.current.drawConnectors(
          landmarks,
          HandLandmarker.HAND_CONNECTIONS,
          { color: "#bef264", lineWidth: 5 } // lime-300
        );
        drawingUtilsRef.current.drawLandmarks(landmarks, { color: "#ec4899", lineWidth: 2 }); // pink-500
      }
    }
    
    canvasCtx.restore();

    // Handle the hand monitor windows
    setDetectedHandsCount(results.landmarks.length);
    const handCanvases = [handCanvasRef1.current, handCanvasRef2.current];
    const handDrawingUtils = [handDrawingUtilsRef1, handDrawingUtilsRef2];

    for (let i = 0; i < handCanvases.length; i++) {
        const handCanvas = handCanvases[i];
        const handData = results.landmarks?.[i];
        
        if (!handCanvas) continue;
        
        const handCtx = handCanvas.getContext("2d");
        if (!handCtx) continue;
        
        if (!handDrawingUtils[i].current) {
            handDrawingUtils[i].current = new DrawingUtils(handCtx);
        }
        const currentDrawingUtils = handDrawingUtils[i].current;

        if (handData) {
            let minX = video.videoWidth, minY = video.videoHeight, maxX = 0, maxY = 0;
            for (const landmark of handData) {
                const x = landmark.x * video.videoWidth;
                const y = landmark.y * video.videoHeight;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }

            const padding = 50;
            const bboxWidth = maxX - minX + 2 * padding;
            const bboxHeight = maxY - minY + 2 * padding;
            const size = Math.max(bboxWidth, bboxHeight);
            const centerX = minX + (maxX - minX) / 2;
            const centerY = minY + (maxY - minY) / 2;
            
            const sx = Math.max(0, centerX - size / 2);
            const sy = Math.max(0, centerY - size / 2);

            handCtx.save();
            handCtx.clearRect(0, 0, handCanvas.width, handCanvas.height);
            handCtx.scale(-1, 1);
            handCtx.translate(-handCanvas.width, 0);
            
            handCtx.drawImage(video, sx, sy, size, size, 0, 0, handCanvas.width, handCanvas.height);
            handCtx.restore();

            const transformedLandmarks = handData.map(lm => {
                const originalX = lm.x * video.videoWidth;
                const originalY = lm.y * video.videoHeight;
                const relativeX = originalX - sx;
                const relativeY = originalY - sy;
                const normalizedX = relativeX / size;
                const normalizedY = relativeY / size;
                return { x: 1.0 - normalizedX, y: normalizedY, z: lm.z };
            });

            currentDrawingUtils?.drawConnectors(transformedLandmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#a3e635", lineWidth: 3 }); // lime-400
            currentDrawingUtils?.drawLandmarks(transformedLandmarks, { color: "#f472b6", lineWidth: 1.5, radius: 3 }); // pink-400
        } else {
            handCtx.clearRect(0, 0, handCanvas.width, handCanvas.height);
        }
    }


    animationFrameId.current = requestAnimationFrame(predictWebcam);
  }, [isWebcamActive]);

  const toggleWebcam = useCallback(async () => {
    if (isWebcamActive) {
      setIsWebcamActive(false);
    } else {
      if (!handLandmarkerRef.current) {
        console.warn("HandLandmarker not ready yet.");
        return;
      }
      setIsWebcamActive(true);
    }
  }, [isWebcamActive]);
  
  useEffect(() => {
    let stream: MediaStream | null = null;
    const video = videoRef.current;

    const onLoadedData = () => {
        if (animationFrameId.current) {
            cancelAnimationFrame(animationFrameId.current);
        }
        animationFrameId.current = requestAnimationFrame(predictWebcam);
    };
    
    const enableWebcam = async () => {
        if (isWebcamActive && video) {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
                video.srcObject = stream;
                video.addEventListener("loadeddata", onLoadedData);
            } catch (err) {
                console.error("Error accessing webcam:", err);
                setError("Could not access webcam. Please check permissions and try again.");
                setIsWebcamActive(false);
            }
        }
    };
    
    const disableWebcam = () => {
        if (video && video.srcObject) {
            const currentStream = video.srcObject as MediaStream;
            currentStream.getTracks().forEach((track) => track.stop());
            video.srcObject = null;
            video.removeEventListener("loadeddata", onLoadedData);
        }
        if (animationFrameId.current) {
            cancelAnimationFrame(animationFrameId.current);
            animationFrameId.current = null;
        }
        const handCtx1 = handCanvasRef1.current?.getContext('2d');
        handCtx1?.clearRect(0,0, handCanvasRef1.current?.width ?? 0, handCanvasRef1.current?.height ?? 0);
        const handCtx2 = handCanvasRef2.current?.getContext('2d');
        handCtx2?.clearRect(0,0, handCanvasRef2.current?.width ?? 0, handCanvasRef2.current?.height ?? 0);
        setDetectedHandsCount(0);
    };

    if(isWebcamActive){
        enableWebcam();
    } else {
        disableWebcam();
    }
    
    return () => {
        disableWebcam();
    };
  }, [isWebcamActive, predictWebcam]);


  return (
    <div className="w-full h-full relative bg-black">
      <header className="absolute top-0 left-0 right-0 z-20 text-center p-4 sm:p-6 lg:p-8 bg-gradient-to-b from-black/70 to-transparent pointer-events-none">
        <h1 
          className="text-4xl sm:text-5xl font-bold text-lime-400 tracking-widest uppercase"
          style={{ textShadow: '0 0 8px #a3e635, 0 0 12px #a3e635' }}
        >
          Fel Hand Tracker
        </h1>
        <p className="mt-2 text-lg text-gray-400 font-sans">
          Channeling the Vision of Sargeras
        </p>
      </header>
      
      <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center">
        {isWebcamActive && <video ref={videoRef} className="w-full h-full object-cover transform -scale-x-100" autoPlay playsInline></video>}
        <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full object-cover transform -scale-x-100"></canvas>
        
        <div className="absolute top-4 left-4 w-[20%] max-w-[200px] flex flex-col gap-4 z-10">
            <canvas 
              ref={handCanvasRef1}
              aria-label="Zoomed view of first detected hand"
              className={`w-full aspect-square border-2 border-lime-400 rounded-lg shadow-[0_0_15px_rgba(132,204,22,0.6)] bg-black/70 backdrop-blur-sm transition-opacity duration-300 ${detectedHandsCount > 0 ? 'opacity-100' : 'opacity-0'}`}
            ></canvas>
             <canvas 
              ref={handCanvasRef2}
              aria-label="Zoomed view of second detected hand"
              className={`w-full aspect-square border-2 border-lime-400 rounded-lg shadow-[0_0_15px_rgba(132,204,22,0.6)] bg-black/70 backdrop-blur-sm transition-opacity duration-300 ${detectedHandsCount > 1 ? 'opacity-100' : 'opacity-0'}`}
            ></canvas>
        </div>

        {!isWebcamActive && !isLoading && (
            <div className="text-center text-lime-800 z-10">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.55a2 2 0 01.45 2.12l-2.5 7A2 2 0 0115.35 21H8.65a2 2 0 01-1.9-1.38l-2.5-7A2 2 0 014.7 10L9 10m0-7a3 3 0 013 3v2m0 0H9m3 0a3 3 0 100-6 3 3 0 000 6z" />
                </svg>
                <p className="font-sans text-lime-600">Vision is dark. Awaken the lens.</p>
            </div>
        )}
      </div>
      
      <footer className="absolute bottom-0 left-0 right-0 z-20 p-4 sm:p-6 lg:p-8 bg-gradient-to-t from-black/70 to-transparent">
        {error && <p className="font-sans text-red-300 bg-red-900/50 border border-red-500/50 p-3 rounded-lg w-full max-w-2xl mx-auto text-center mb-4">{error}</p>}

        <div className="w-full flex flex-col justify-center items-center text-center">
          {isLoading ? (
            <div className="flex items-center space-x-3 text-lime-400">
              <svg className="animate-spin h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span className="text-lg uppercase tracking-wider">Summoning Model...</span>
            </div>
          ) : (
            <button
              onClick={toggleWebcam}
              disabled={isLoading}
              className={`px-8 py-3 text-lg font-bold uppercase tracking-widest rounded-md transition-all duration-300 ease-in-out transform hover:scale-105 focus:outline-none focus:ring-4 focus:ring-opacity-50 ${
                isWebcamActive
                  ? "border-2 border-purple-500 bg-transparent text-purple-400 hover:bg-purple-500 hover:text-white hover:shadow-[0_0_20px_#a855f7] focus:ring-purple-400"
                  : "border-2 border-lime-500 bg-transparent text-lime-400 hover:bg-lime-500 hover:text-black hover:shadow-[0_0_20px_#84cc16] focus:ring-lime-400"
              } shadow-lg`}
            >
              {isWebcamActive ? "End Vision" : "Begin Ritual"}
            </button>
          )}
           <p className="w-full max-w-5xl text-center mt-4 text-gray-600 text-sm font-sans">Your webcam feed is processed locally. Not even the Burning Legion sees your data.</p>
        </div>
      </footer>
    </div>
  );
};

export default HandDetector;
