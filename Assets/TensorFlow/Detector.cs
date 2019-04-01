using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TensorFlow;

public enum DetectionModels
{
    YOLO,
    SSD
}

public class Detector : MonoBehaviour
{
    readonly double[] ANCHORS = {
        0.57273,
        0.677385,
        1.87446,
        2.06253,
        3.33843,
        5.47434,
        7.88282,
        3.52778,
        9.77052,
        9.16828
    };

    TFGraph graph;
    static TFSession session;
    string[] labels;

    readonly string _inputName;
    readonly string _outputName;
    readonly int _inputHeight;
    readonly int _inputWidth;
    readonly float _inputMean;
    readonly float _inputStd;
    readonly double[] _anchors;
    readonly int _blockSize;
    readonly int _numBoxesPerBlock;
    readonly DetectionModels _detectionModel;

    public Detector(TextAsset modelFile,
                    TextAsset labelFile,
                    DetectionModels model = DetectionModels.SSD,
                    string input = "input",
                    string output = "output",
                    int height = 300,
                    int width = 300,
                    float mean = 127.5f,
                    float std = 127.5f,
                    int blockSize = 32,
                    int numBoxesPerBlock = 5,
                    double[] anchors = null)
    {

#if UNITY_ANDROID
        TensorFlowSharp.Android.NativeBinding.Init ();
#endif
        graph = new TFGraph();
        graph.Import(modelFile.bytes);
        session = new TFSession(graph);

        labels = labelFile.text.Split(new char[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);

        _detectionModel = model;
        _inputName = input;
        _outputName = output;
        _inputHeight = height;
        _inputWidth = width;
        _inputMean = mean;
        _inputStd = std;
        _anchors = anchors ?? ANCHORS;
        _blockSize = blockSize;
        _numBoxesPerBlock = numBoxesPerBlock;
    }

    public void Close()
    {
        session?.Dispose();
        graph?.Dispose();
        labels = null;
    }

    public IList Detect(Texture2D texture, int numResultsPerClass = 1, float threshold = 0.2f, 
                        int angle = 0, Flip flip = Flip.NONE)
    {
        var shape = new TFShape(1, _inputWidth, _inputHeight, 3);
        var input = graph[_inputName][0];
        TFTensor inputTensor = null;

        if (input.OutputType == TFDataType.Float)
        {
            float[] imgData = Utils.DecodeTexture(texture, _inputWidth, _inputHeight,
                                                  _inputMean, _inputStd, angle, flip);
            inputTensor = TFTensor.FromBuffer(shape, imgData, 0, imgData.Length);
        }
        else if (input.OutputType == TFDataType.UInt8)
        {
            byte[] imgData = Utils.DecodeTexture(texture, _inputWidth, _inputHeight, angle, flip);
            inputTensor = TFTensor.FromBuffer(shape, imgData, 0, imgData.Length);
        }
        else
        {
            throw new Exception($"Input date type {input.OutputType} is not supported.");
        }

        var runner = session.GetRunner();
        runner.AddInput(input, inputTensor);

        IList results;

        if (_detectionModel == DetectionModels.SSD)
        {
            results = ParseSSD(runner, threshold, numResultsPerClass);
        }
        else
        {
            results = ParseYOLO(runner, threshold, numResultsPerClass);
        }

        inputTensor.Dispose();

        return results;
    }

    private IList ParseSSD(TFSession.Runner runner, float threshold, int numResultsPerClass)
    {
        runner.Fetch(graph["detection_boxes"][0],
                     graph["detection_classes"][0],
                     graph["detection_scores"][0],
                     graph["num_detections"][0]);

        var outputs = runner.Run();

        var boxes = outputs[0].GetValue() as float[,,];
        var classes = outputs[1].GetValue() as float[,];
        var scores = outputs[2].GetValue() as float[,];
        var num_detections = outputs[3].GetValue() as float[];

        foreach (var o in outputs) o.Dispose();

        var results = new List<Dictionary<string, object>>();
        var counters = new Dictionary<string, int>();

        for (int i = 0; i < (int)num_detections[0]; i++)
        {
            if (scores[0, i] < threshold) continue;

            string detectedClass = labels[(int)classes[0, i]];

            if (counters.ContainsKey(detectedClass))
            {
                if (counters[detectedClass] >= numResultsPerClass) continue;
                counters[detectedClass] += 1;
            }
            else
            {
                counters.Add(detectedClass, 1);
            }

            float ymin = Math.Max(0, boxes[0, i, 0]);
            float xmin = Math.Max(0, boxes[0, i, 1]);
            float ymax = boxes[0, i, 2];
            float xmax = boxes[0, i, 3];

            var rect = new Dictionary<string, float>
                {
                    { "x", xmin },
                    { "y", ymin },
                    { "w", Math.Min(1 - xmin, xmax - xmin) },
                    { "h", Math.Min(1 - ymin, ymax - ymin) }
                };

            var result = new Dictionary<string, object>
                {
                    { "rect", rect },
                    { "confidenceInClass", scores[0, i] },
                    { "detectedClass", detectedClass }
                };

            results.Add(result);
        }

        //Utils.Log(results);

        return results;
    }

    private IList ParseYOLO(TFSession.Runner runner, float threshold, int numResultsPerClass)
    {
        runner.Fetch(graph["output"][0]);
        var outputs = runner.Run();
        var output = outputs[0].GetValue() as float[,,,];

        foreach (var o in outputs) o.Dispose();

        var gridSize = _inputWidth / _blockSize;
        int numClasses = labels.Length;

        var list = new List<Dictionary<string, object>>();

        for (int y = 0; y < gridSize; y++)
        {
            for (int x = 0; x < gridSize; x++)
            {
                for (int b = 0; b < _numBoxesPerBlock; b++)
                {
                    int offset = (numClasses + 5) * b;

                    float confidence = expit(output[0, y, x, offset + 4]);

                    float[] classes = new float[numClasses];
                    for (int c = 0; c < numClasses; c++)
                    {
                        classes[c] = output[0, y, x, offset + 5 + c];
                    }
                    softmax(classes);

                    int detectedClass = -1;
                    float maxClass = 0;
                    for (int c = 0; c < numClasses; ++c)
                    {
                        if (classes[c] > maxClass)
                        {
                            detectedClass = c;
                            maxClass = classes[c];
                        }
                    }

                    float confidenceInClass = maxClass * confidence;

                    if(confidenceInClass > threshold) {
                        float xPos = (x + expit(output[0, y, x, offset + 0])) * _blockSize;
                        float yPos = (y + expit(output[0, y, x, offset + 1])) * _blockSize;

                        float w = (float)((Math.Exp(output[0, y, x, offset + 2]) * _anchors[2 * b + 0]) * _blockSize);
                        float h = (float)(Math.Exp(output[0, y, x, offset + 3]) * _anchors[2 * b + 1]) * _blockSize;

                        float xmin = Math.Max(0, (xPos - w / 2) / _inputWidth);
                        float ymin = Math.Max(0, (yPos - h / 2) / _inputHeight);

                        var rect = new Dictionary<string, float>
                            {
                                { "x", xmin },
                                { "y", ymin },
                                { "w", Math.Min(1 - xmin, w / _inputWidth) },
                                { "h", Math.Min(1 - ymin, h / _inputHeight) }
                            };

                        var result = new Dictionary<string, object>
                            {
                                { "rect", rect },
                                { "confidenceInClass", confidenceInClass },
                                { "detectedClass", labels[detectedClass] }
                            };

                        list.Add(result);
                    }
                }

            }
        }

        var sortedList = list.OrderByDescending(i => i["confidenceInClass"]).ToList();

        var results = new List<Dictionary<string, object>>();
        var counters = new Dictionary<string, int>();
        sortedList.ForEach(i =>
        {
            String detectedClass = (string)i["detectedClass"];

            if (counters.ContainsKey(detectedClass))
            {
                if (counters[detectedClass] >= numResultsPerClass) return;
                counters[detectedClass] += 1;
            }
            else
            {
                counters.Add(detectedClass, 1);
            }

            results.Add(i);
        });

        //Utils.Log(results);

        return results;
    }

    private float expit(float x)
    {
        return (float)(1.0 / (1.0 + Math.Exp(-x)));
    }

    private void softmax(float[] vals)
    {
        float max = float.NegativeInfinity;
        foreach (float val in vals)
        {
            max = Math.Max(max, val);
        }
        float sum = 0.0f;
        for (int i = 0; i < vals.Length; ++i)
        {
            vals[i] = (float)Math.Exp(vals[i] - max);
            sum += vals[i];
        }
        for (int i = 0; i < vals.Length; ++i)
        {
            vals[i] = vals[i] / sum;
        }
    }
}
