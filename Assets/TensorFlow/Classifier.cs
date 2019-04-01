using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TensorFlow;

public class Classifier
{
    TFGraph graph;
    TFSession session;
    string[] labels;

    readonly string inputName;
    readonly string outputName;
    readonly int inputHeight;
    readonly int inputWidth;
    readonly float inputMean;
    readonly float inputStd;

    public Classifier(TextAsset modelFile,
                      TextAsset labelFile,
                      string input = "input",
                      string output = "output",
                      int height = 224,
                      int width = 224,
                      float mean = 127.5f,
                      float std = 127.5f)
    {
#if UNITY_ANDROID
        TensorFlowSharp.Android.NativeBinding.Init ();
#endif
        graph = new TFGraph();
        graph.Import(modelFile.bytes);
        session = new TFSession(graph);

        labels = labelFile.text.Split(new char[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);

        inputName = input;
        outputName = output;
        inputHeight = height;
        inputWidth = width;
        inputMean = mean;
        inputStd = std;
    }

    public void Close()
    {
        session?.Dispose();
        graph?.Dispose();
        labels = null;
    }

    public IList Classify(Texture2D texture, int numResults = 5, float threshold = 0.1f, 
                          int angle = 0, Flip flip = Flip.NONE)
    {
        var shape = new TFShape(1, inputWidth, inputHeight, 3);
        var input = graph[inputName][0];
        TFTensor inputTensor = null;

        if (input.OutputType == TFDataType.Float)
        {
            float[] imgData = Utils.DecodeTexture(texture, inputWidth, inputHeight, 
                                                  inputMean, inputStd, angle, flip);
            inputTensor = TFTensor.FromBuffer(shape, imgData, 0, imgData.Length);
        }
        else if (input.OutputType == TFDataType.UInt8)
        {
            byte[] imgData = Utils.DecodeTexture(texture, inputWidth, inputHeight, angle, flip);
            inputTensor = TFTensor.FromBuffer(shape, imgData, 0, imgData.Length);
        }
        else
        {
            throw new Exception($"Input date type {input.OutputType} is not supported.");
        }

        var runner = session.GetRunner();
        runner.AddInput(input, inputTensor).Fetch(graph[outputName][0]);

        var output = runner.Run()[0];
        var outputs = output.GetValue() as float[,];

        inputTensor.Dispose();
        output.Dispose();

        var list = new List<KeyValuePair<string, float>>();

        for (int i = 0; i < labels.Length; i++)
        {
            var confidence = outputs[0, i];
            if (confidence < threshold) continue;

            list.Add(new KeyValuePair<string, float>(labels[i], confidence));
        }

        var results = list.OrderByDescending(i => i.Value).Take(numResults).ToList();

        //Utils.Log(results);

        return results;
    }
}
