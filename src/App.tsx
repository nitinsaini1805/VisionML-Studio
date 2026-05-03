import React, { useRef, useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import Webcam from 'react-webcam';
import { Plus, Upload, Trash2, Camera, Brain, Activity, Download, Settings2, X, Image as ImageIcon } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { cn } from './lib/utils';
import { useClassifier } from './hooks/useClassifier';

function Card({ className, children }: { className?: string; children: React.ReactNode }) {
  return <div className={cn("bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden", className)}>{children}</div>;
}

export default function App() {
  const {
    isReady,
    isTraining,
    classes,
    logs,
    accuracy,
    addClass,
    addExample,
    clearData,
    train,
    predict,
    predictPretrained,
    downloadModel
  } = useClassifier();

  const [newClassName, setNewClassName] = useState('');
  const [activeTab, setActiveTab] = useState<'quick' | 'train' | 'predict'>('quick');
  
  // Hyperparameters
  const [epochs, setEpochs] = useState<number>(20);
  const [batchSize, setBatchSize] = useState<number>(16);
  const [learningRate, setLearningRate] = useState<number>(0.0001);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Prediction State
  const [predictionImg, setPredictionImg] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<{className: string; probability: number}[] | null>(null);
  const predictImgRef = useRef<HTMLImageElement>(null);
  
  // Webcam Predict State
  const predictWebcamRef = useRef<Webcam>(null);
  const [isPredictingWebcam, setIsPredictingWebcam] = useState(false);

  const handleAddClass = (e: React.FormEvent) => {
    e.preventDefault();
    if (newClassName.trim()) {
      addClass(newClassName.trim());
      setNewClassName('');
    }
  };

  const handleTrain = () => {
    train(epochs, batchSize, learningRate).catch(err => alert(err.message));
  };

  const onPredictDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;
    const file = acceptedFiles[0];
    const url = URL.createObjectURL(file);
    setIsPredictingWebcam(false);
    setPredictionImg(url);
    setPredictions(null);
  }, []);

  const { getRootProps: getPredictRootProps, getInputProps: getPredictInputProps, isDragActive: isPredictDragActive } = useDropzone({
    onDrop: onPredictDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png'] } as any,
    maxFiles: 1
  } as any);

  const performPrediction = async () => {
    if (predictImgRef.current) {
      if (activeTab === 'quick') {
        const res = await predictPretrained(predictImgRef.current);
        setPredictions(res);
      } else {
        const res = await predict(predictImgRef.current);
        setPredictions(res);
      }
    }
  };

  const captureAndPredict = useCallback(async () => {
    if (predictWebcamRef.current) {
      const imageSrc = predictWebcamRef.current.getScreenshot();
      if (imageSrc) {
        setPredictionImg(imageSrc);
        setIsPredictingWebcam(false);
        setPredictions(null); // will trigger performPrediction on load
      }
    }
  }, []);

  if (!isReady) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center flex-col space-y-4">
        <Activity className="w-12 h-12 text-blue-600 animate-pulse" />
        <p className="text-gray-600 font-medium">Loading Deep Learning Models...</p>
      </div>
    );
  }

  const totalExamples = classes.reduce((sum, c) => sum + c.examples, 0);
  const canTrain = classes.length >= 2 && totalExamples >= classes.length * 5;

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 font-sans pb-12">
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-gray-900">VisionML Studio</h1>
          </div>
          <div className="flex bg-gray-100 p-1 rounded-lg">
            <button
              onClick={() => setActiveTab('quick')}
              className={cn("px-4 py-1.5 text-sm font-medium rounded-md transition-colors", activeTab === 'quick' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500 hover:text-gray-700')}
            >
              1. Quick Test
            </button>
            <button
              onClick={() => setActiveTab('train')}
              className={cn("px-4 py-1.5 text-sm font-medium rounded-md transition-colors", activeTab === 'train' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500 hover:text-gray-700')}
            >
              2. Custom Train
            </button>
            <button
              onClick={() => setActiveTab('predict')}
              className={cn("px-4 py-1.5 text-sm font-medium rounded-md transition-colors", activeTab === 'predict' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500 hover:text-gray-700')}
            >
              3. Predict
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {activeTab === 'train' ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="col-span-1 lg:col-span-1 space-y-6">
               <div className="space-y-2">
                 <h2 className="text-lg font-semibold">1. Define Categories</h2>
                 <p className="text-sm text-gray-500 bg-blue-50 p-3 rounded-lg border border-blue-100 text-blue-800">
                    Add classes, then use your webcam or drop images to build a dataset. 
                    Aim for at least 20 images per category.
                 </p>
               </div>
               
               <form onSubmit={handleAddClass} className="flex gap-2">
                 <input
                   type="text"
                   value={newClassName}
                   onChange={e => setNewClassName(e.target.value)}
                   placeholder="e.g. Happy Face"
                   className="flex-1 bg-white border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                 />
                 <button disabled={isTraining || newClassName.trim().length === 0} type="submit" className="bg-gray-900 disabled:opacity-50 text-white rounded-lg px-4 py-2 text-sm font-medium hover:bg-gray-800 transition-colors flex items-center">
                   <Plus className="w-4 h-4 mr-1" /> Add
                 </button>
               </form>

               <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2 pb-4">
                  {classes.map((c, i) => (
                    <ClassItem key={c.id} classData={c} index={i} addExample={addExample} isTraining={isTraining} />
                  ))}
               </div>

               {classes.length > 0 && (
                 <div className="pt-4 border-t border-gray-200">
                    <button onClick={clearData} disabled={isTraining} className="w-full py-2 bg-red-50 text-sm text-red-600 font-medium rounded-lg hover:bg-red-100 disabled:opacity-50 transition-colors">
                      Clear all data
                    </button>
                 </div>
               )}
            </div>

            <div className="col-span-1 lg:col-span-2 space-y-6">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-white p-4 rounded-xl border border-gray-200 shadow-sm">
                <div className="space-y-1">
                  <h2 className="text-lg font-semibold">2. Training Dashboard</h2>
                  <p className="text-sm text-gray-500">Train your CNN transfer learning model.</p>
                </div>
                
                <div className="flex flex-col sm:flex-row items-center gap-3">
                  <button
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="text-sm font-medium text-gray-600 hover:text-gray-900 flex items-center px-3 py-2 rounded-lg hover:bg-gray-100 transition-colors"
                  >
                    <Settings2 className="w-4 h-4 mr-2" />
                    Hyperparameters
                  </button>
                  <button
                    onClick={handleTrain}
                    disabled={!canTrain || isTraining}
                    className="bg-blue-600 disabled:bg-gray-300 text-white px-6 py-2 rounded-lg font-medium shadow-sm hover:bg-blue-700 transition-colors flex items-center space-x-2 w-full sm:w-auto justify-center"
                  >
                    {isTraining ? (
                       <><Activity className="w-4 h-4 animate-spin" /><span>Training...</span></>
                    ) : (
                       <><Brain className="w-4 h-4" /><span>Start Training</span></>
                    )}
                  </button>
                </div>
              </div>

              {showAdvanced && (
                <div className="bg-gray-50 p-4 border border-gray-200 rounded-xl grid grid-cols-1 sm:grid-cols-3 gap-4 animate-in slide-in-from-top-2">
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-gray-600 uppercase">Epochs</label>
                    <input 
                      type="number" min="1" max="200" 
                      value={epochs} onChange={e => setEpochs(Number(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm bg-white" disabled={isTraining}
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-gray-600 uppercase">Batch Size</label>
                    <select 
                      value={batchSize} onChange={e => setBatchSize(Number(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm bg-white" disabled={isTraining}
                    >
                      <option value={8}>8</option>
                      <option value={16}>16</option>
                      <option value={32}>32</option>
                      <option value={64}>64</option>
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-gray-600 uppercase">Learning Rate</label>
                    <select 
                      value={learningRate} onChange={e => setLearningRate(Number(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm bg-white" disabled={isTraining}
                    >
                      <option value={0.001}>0.001</option>
                      <option value={0.0001}>0.0001</option>
                      <option value={0.00001}>0.00001</option>
                    </select>
                  </div>
                </div>
              )}

              {!canTrain && classes.length > 0 && (
                <div className="bg-yellow-50 text-yellow-800 p-4 rounded-lg text-sm border border-yellow-200 flex items-start">
                  <span className="mr-2 text-base">⚠️</span> 
                  <span>Add at least 2 categories and minimum 5 images per category to start training. More images = better accuracy!</span>
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card className="p-4 flex flex-col items-center justify-center min-h-[250px]">
                  <h3 className="text-sm font-semibold text-gray-700 mb-4 self-start">Accuracy</h3>
                  {logs.length > 0 ? (
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={logs}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="epoch" tick={{fontSize: 12}} />
                        <YAxis domain={[0, 1]} tick={{fontSize: 12}} />
                        <Tooltip />
                        <Legend iconType="circle" wrapperStyle={{fontSize: "12px"}}/>
                        <Line type="monotone" dataKey="acc" stroke="#2563eb" name="Train Acc" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="val_acc" stroke="#16a34a" name="Val Acc" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <p className="text-gray-400 text-sm">Waiting for dataset...</p>
                  )}
                </Card>

                <Card className="p-4 flex flex-col items-center justify-center min-h-[250px]">
                  <h3 className="text-sm font-semibold text-gray-700 mb-4 self-start">Loss</h3>
                  {logs.length > 0 ? (
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={logs}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="epoch" tick={{fontSize: 12}} />
                        <YAxis tick={{fontSize: 12}} />
                        <Tooltip />
                        <Legend iconType="circle" wrapperStyle={{fontSize: "12px"}}/>
                        <Line type="monotone" dataKey="loss" stroke="#ef4444" name="Train Loss" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="val_loss" stroke="#f97316" name="Val Loss" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <p className="text-gray-400 text-sm">Waiting for dataset...</p>
                  )}
                </Card>
              </div>
              
              {accuracy !== null && !isTraining && (
                <div className="bg-green-50 border border-green-200 text-green-800 p-6 rounded-xl flex flex-col md:flex-row items-center justify-between gap-4">
                  <div className="flex-1">
                    <h3 className="font-semibold text-green-900 text-lg mb-1">Model Training Complete!</h3>
                    <p className="text-sm text-green-700">Your model achieved a validation accuracy of <span className="font-bold text-green-900">{(accuracy * 100).toFixed(1)}%</span>.</p>
                  </div>
                  <div className="flex items-center gap-3 w-full md:w-auto">
                    <button 
                      onClick={downloadModel}
                      className="flex-1 md:flex-none justify-center px-4 py-2 bg-white text-green-800 border border-green-300 hover:bg-green-100 rounded-lg font-medium transition-colors text-sm flex items-center shadow-sm"
                    >
                      <Download className="w-4 h-4 mr-2" /> Download Model
                    </button>
                    <button 
                      onClick={() => setActiveTab('predict')}
                      className="flex-1 md:flex-none justify-center bg-green-600 text-white px-6 py-2 rounded-lg font-medium shadow-sm hover:bg-green-700 transition-colors text-sm"
                    >
                      Test Model ➔
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto space-y-8">
             <div className="text-center space-y-2">
                <h2 className="text-2xl font-bold text-gray-900">
                  {activeTab === 'quick' ? 'Quick Predict (Pre-trained MobileNet)' : 'Test Your Custom Model'}
                </h2>
                <p className="text-gray-500">
                  {activeTab === 'quick' 
                    ? 'Instantly classify general objects using the 1000 original ImageNet classes.' 
                    : 'Feed new unseen images into the model you just trained to see how well it performs.'}
                </p>
             </div>

             {activeTab === 'predict' && !accuracy ? (
               <div className="bg-yellow-50 text-yellow-800 p-6 rounded-xl border border-yellow-200 text-center shadow-sm">
                 <div className="w-12 h-12 bg-yellow-100 rounded-full flex items-center justify-center mx-auto mb-4">
                   <Brain className="w-6 h-6 text-yellow-600" />
                 </div>
                 <h3 className="font-semibold text-lg text-yellow-900 mb-2">No Model Trained Yet</h3>
                 <p className="text-yellow-700 mb-6">You need to train your custom model before you can test it.</p>
                 <button onClick={() => setActiveTab('train')} className="px-6 py-2.5 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg font-medium transition-colors text-sm shadow-sm">
                   Go to Training Setup
                 </button>
               </div>
             ) : (
               <div className="bg-white border text-center border-gray-200 rounded-2xl p-6 sm:p-10 shadow-sm">
                  
                  <div className="flex justify-center mb-6 space-x-2">
                    <button
                      onClick={() => { setIsPredictingWebcam(false); setPredictionImg(null); }}
                      className={cn("px-4 py-2 text-sm font-medium rounded-lg transition-colors flex items-center", !isPredictingWebcam ? "bg-blue-50 text-blue-700" : "bg-gray-50 text-gray-600 hover:bg-gray-100")}
                    >
                      <ImageIcon className="w-4 h-4 mr-2" /> File Upload
                    </button>
                    <button
                      onClick={() => { setIsPredictingWebcam(true); setPredictionImg(null); }}
                      className={cn("px-4 py-2 text-sm font-medium rounded-lg transition-colors flex items-center", isPredictingWebcam ? "bg-blue-50 text-blue-700" : "bg-gray-50 text-gray-600 hover:bg-gray-100")}
                    >
                      <Camera className="w-4 h-4 mr-2" /> Live Camera
                    </button>
                  </div>

                  {isPredictingWebcam ? (
                    <div className="relative border border-gray-200 rounded-xl overflow-hidden bg-gray-50 aspect-video flex items-center justify-center mb-8">
                       <Webcam
                         {...{
                           audio: false,
                           ref: predictWebcamRef,
                           screenshotFormat: "image/jpeg",
                           videoConstraints: { facingMode: "environment" },
                           className: "w-full h-full object-cover"
                         } as any}
                       />
                       <button 
                         onClick={captureAndPredict} 
                         className="absolute bottom-6 left-1/2 -translate-x-1/2 bg-blue-600 text-white px-6 py-2.5 rounded-full font-semibold shadow-lg hover:bg-blue-700 flex items-center"
                       >
                         <Camera className="w-4 h-4 mr-2" /> Capture to Predict
                       </button>
                    </div>
                  ) : (
                    <div 
                      {...getPredictRootProps()} 
                      className={cn(
                        "border-2 border-dashed rounded-xl p-8 transition-all cursor-pointer mb-8",
                        isPredictDragActive ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-gray-400 hover:bg-gray-50",
                        predictionImg ? "p-4 border-none bg-gray-50" : ""
                      )}
                    >
                      <input {...getPredictInputProps()} />
                      {predictionImg ? (
                        <div className="relative inline-block w-full max-w-sm mx-auto">
                          <img 
                             ref={predictImgRef}
                             src={predictionImg} 
                             alt="To predict" 
                             className="w-full rounded-lg shadow-md object-contain max-h-80 mx-auto"
                             crossOrigin="anonymous"
                             onLoad={performPrediction}
                          />
                          <button 
                             onClick={(e) => { e.stopPropagation(); setPredictionImg(null); setPredictions(null); }}
                             className="absolute -top-3 -right-3 bg-red-500 text-white p-1.5 rounded-full shadow-md hover:bg-red-600"
                          >
                             <X className="w-4 h-4"/>
                          </button>
                        </div>
                      ) : (
                        <div className="flex flex-col items-center justify-center space-y-3 py-10">
                          <div className="w-14 h-14 bg-blue-50 rounded-full flex items-center justify-center text-blue-600">
                            <Upload className="w-7 h-7" />
                          </div>
                          <div className="text-gray-700 text-lg">
                            <span className="font-semibold text-blue-600">Click to upload</span> or drag and drop
                          </div>
                          <p className="text-sm text-gray-500">Use any image (JPG, PNG)</p>
                        </div>
                      )}
                    </div>
                  )}

                  {predictions && predictionImg && (
                    <div className="space-y-4 text-left max-w-md mx-auto bg-gray-50 p-6 rounded-xl border border-gray-100">
                      <h4 className="font-semibold text-gray-900 border-b pb-3 mb-4">Classification Results</h4>
                      <div className="space-y-3">
                        {predictions.slice(0, 3).map((p, i) => (
                          <div key={i} className={cn("p-4 rounded-lg border", i === 0 ? "bg-white border-blue-200 shadow-sm" : "bg-transparent border-transparent")}>
                             <div className="flex justify-between items-center mb-1.5">
                               <span className={cn("font-medium truncate pr-4", i === 0 ? "text-blue-800 text-lg" : "text-gray-700")}>
                                 {p.className}
                               </span>
                               <span className={cn("font-bold", i === 0 ? "text-blue-700 text-lg" : "text-gray-500 text-sm")}>
                                 {(p.probability * 100).toFixed(1)}%
                               </span>
                             </div>
                             <div className="w-full bg-gray-200 rounded-full h-2">
                               <div 
                                 className={cn("h-2 rounded-full", i === 0 ? "bg-blue-600" : "bg-gray-400")} 
                                 style={{ width: `${Math.max(p.probability * 100, 2)}%` }}
                               ></div>
                             </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
               </div>
             )}
          </div>
        )}
      </main>
    </div>
  );
}

const ClassItem: React.FC<{ 
  classData: any, 
  index: number, 
  addExample: (img: HTMLImageElement, idx: number) => Promise<void>,
  isTraining: boolean
}> = ({ 
  classData, 
  index, 
  addExample,
  isTraining
}) => {
  const [usingWebcam, setUsingWebcam] = useState(false);
  const webcamRef = useRef<Webcam>(null);

  const capture = useCallback((e?: React.MouseEvent) => {
    if (e) e.preventDefault();
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        const img = new Image();
        img.src = imageSrc;
        img.onload = () => {
          addExample(img, index);
        };
      }
    }
  }, [addExample, index]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    acceptedFiles.forEach(file => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.src = url;
      img.onload = () => {
        addExample(img, index);
      };
    });
  }, [addExample, index]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png'] } as any,
    disabled: isTraining || usingWebcam
  } as any);

  return (
    <Card className="flex flex-col">
      <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between bg-gray-50/50">
        <span className="font-semibold text-gray-800 text-sm">{classData.name}</span>
        <span className="bg-blue-100 text-blue-800 text-xs py-1 px-2.5 rounded-full font-bold shadow-sm">
          {classData.examples} img
        </span>
      </div>
      <div className="p-3 flex gap-2 w-full h-[140px]">
        {usingWebcam ? (
          <div className="relative flex-1 bg-black rounded-lg overflow-hidden h-full">
            <Webcam
              {...{
                audio: false,
                ref: webcamRef,
                screenshotFormat: "image/jpeg",
                videoConstraints: { facingMode: "environment" },
                className: "w-full h-full object-cover"
              } as any}
            />
            <button 
              onClick={capture} 
              className="absolute bottom-2 left-1/2 -translate-x-1/2 bg-white text-gray-900 px-4 py-1.5 rounded-full text-xs font-bold shadow-md hover:bg-gray-100 active:scale-95 transition-transform"
            >
              Hold / Click
            </button>
            <button 
              onClick={() => setUsingWebcam(false)} 
              className="absolute top-2 right-2 bg-black/50 text-white p-1 rounded-full hover:bg-black/70"
            >
              <X className="w-4 h-4"/>
            </button>
          </div>
        ) : (
          <>
            <div 
              {...getRootProps()} 
              className={cn(
                "flex-1 border-2 border-dashed rounded-lg text-center cursor-pointer transition-colors flex flex-col items-center justify-center h-full",
                isDragActive ? "border-blue-400 bg-blue-50" : "border-gray-200 hover:border-gray-300 hover:bg-gray-50",
                isTraining ? "opacity-50 cursor-not-allowed" : ""
              )}
            >
              <input {...getInputProps()} />
              <Upload className="mx-auto h-5 w-5 text-gray-400 mb-2" />
              <p className="text-xs text-gray-500 font-medium">Drop Images</p>
            </div>
            <button 
              onClick={() => setUsingWebcam(true)} 
              disabled={isTraining}
              className="w-[80px] flex-none border border-gray-200 rounded-lg bg-gray-50 hover:bg-gray-100 flex flex-col items-center justify-center transition-colors disabled:opacity-50 h-full"
            >
              <Camera className="h-6 w-6 text-gray-500 mb-1" />
              <span className="text-[10px] font-medium text-gray-500 uppercase tracking-wider">Webcam</span>
            </button>
          </>
        )}
      </div>
    </Card>
  );
}
