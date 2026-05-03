import { useState, useRef, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';

export interface ClassData {
  id: string;
  name: string;
  examples: number;
}

export interface TrainingLog {
  epoch: number;
  loss: number;
  acc: number;
  val_loss?: number;
  val_acc?: number;
}

export function useClassifier() {
  const [isReady, setIsReady] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [classes, setClasses] = useState<ClassData[]>([]);
  const [logs, setLogs] = useState<TrainingLog[]>([]);
  const [accuracy, setAccuracy] = useState<number | null>(null);

  const mobilenetRef = useRef<mobilenet.MobileNet | null>(null);
  const modelRef = useRef<tf.Sequential | null>(null);

  // Store feature vectors and labels
  const featuresRef = useRef<{ xs: tf.Tensor | null; ys: number[] }>({ xs: null, ys: [] });

  useEffect(() => {
    // Initialize MobileNet
    async function init() {
      await tf.ready();
      // Load mobilenet with smaller alpha for faster performance in browser
      mobilenetRef.current = await mobilenet.load({ version: 2, alpha: 0.5 });
      setIsReady(true);
    }
    init();
  }, []);

  const addClass = useCallback((name: string) => {
    const id = Date.now().toString();
    setClasses((prev) => [...prev, { id, name, examples: 0 }]);
  }, []);

  const clearData = useCallback(() => {
    setClasses([]);
    setLogs([]);
    setAccuracy(null);
    if (featuresRef.current.xs) {
      featuresRef.current.xs.dispose();
    }
    featuresRef.current = { xs: null, ys: [] };
    if (modelRef.current) {
      modelRef.current.dispose();
      modelRef.current = null;
    }
  }, []);

  const addExample = useCallback(async (imageElement: HTMLImageElement, classIndex: number) => {
    if (!mobilenetRef.current) return;

    // Use MobileNet to get the intermediate feature vector
    const feature = mobilenetRef.current.infer(imageElement, true);
    
    const { xs, ys } = featuresRef.current;
    
    if (xs == null) {
      featuresRef.current.xs = tf.keep(feature);
    } else {
      const oldX = xs;
      featuresRef.current.xs = tf.keep(oldX.concat(feature, 0));
      oldX.dispose();
      feature.dispose();
    }

    featuresRef.current.ys.push(classIndex);
    
    setClasses(prev => {
      const newClasses = [...prev];
      newClasses[classIndex] = { ...newClasses[classIndex], examples: newClasses[classIndex].examples + 1 };
      return newClasses;
    });
  }, []);

  const train = useCallback(async (epochs = 30, batchSize = 16, learningRate = 0.0001) => {
    const { xs, ys } = featuresRef.current;
    if (!xs || ys.length === 0 || classes.length < 2) {
      throw new Error("Add more examples to at least 2 classes before training.");
    }

    setIsTraining(true);
    setLogs([]);
    setAccuracy(null);

    // Create the smaller head model
    if (modelRef.current) {
      modelRef.current.dispose();
    }

    const numClasses = classes.length;
    
    const model = tf.sequential({
      layers: [
        tf.layers.flatten({ inputShape: mobilenetRef.current?.outputs[0].shape.slice(1) }),
        tf.layers.dense({ units: 100, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.25 }),
        tf.layers.dense({ units: numClasses, activation: 'softmax' })
      ]
    });

    const optimizer = tf.train.adam(learningRate);
    model.compile({ optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    // Prepare labels
    const ysTensor = tf.oneHot(tf.tensor1d(ys, 'int32'), numClasses);

    // If we have enough examples, split for validation
    const validationSplit = ys.length > 10 ? 0.2 : 0;

    await model.fit(xs, ysTensor, {
      batchSize,
      epochs,
      validationSplit,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (logs) {
            setLogs(prev => [...prev, {
              epoch: epoch + 1,
              loss: logs.loss,
              acc: logs.acc,
              val_loss: logs.val_loss,
              val_acc: logs.val_acc,
            }]);
          }
          // Force UI update
          return tf.nextFrame();
        }
      }
    });

    modelRef.current = model;
    
    // Set final accuracy from last log
    setLogs(prev => {
      if (prev.length > 0) {
        setAccuracy(prev[prev.length - 1].val_acc ?? prev[prev.length - 1].acc);
      }
      return prev;
    });
    
    ysTensor.dispose();
    setIsTraining(false);
  }, [classes.length]);

  const predictPretrained = useCallback(async (imageElement: HTMLImageElement) => {
    if (!mobilenetRef.current) return null;
    const results = await mobilenetRef.current.classify(imageElement);
    return results.map(r => ({
      className: r.className,
      probability: r.probability
    }));
  }, []);

  const predict = useCallback(async (imageElement: HTMLImageElement) => {
    if (!mobilenetRef.current || !modelRef.current) return null;

    const feature = mobilenetRef.current.infer(imageElement, true);
    const predictionTensor = modelRef.current.predict(feature) as tf.Tensor;
    const predictionData = await predictionTensor.data();
    
    feature.dispose();
    predictionTensor.dispose();

    const probabilities = Array.from(predictionData);
    
    return classes.map((c, i) => ({
      className: c.name,
      probability: probabilities[i]
    })).sort((a, b) => b.probability - a.probability);
    
  }, [classes]);

  const downloadModel = useCallback(async () => {
    if (modelRef.current) {
      await modelRef.current.save('downloads://custom-cnn-model');
    }
  }, []);

  return {
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
  };
}
