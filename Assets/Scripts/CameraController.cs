using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Experimental.XR;
using UnityEngine.UI;
using UnityEngine.XR.ARExtensions;
using UnityEngine.XR.ARFoundation;

/// <summary>
/// This component tests getting the latest camera image
/// and converting it to RGBA format. If successful,
/// it displays the image on the screen as a RawImage
/// and also displays information about the image.
/// 
/// This is useful for computer vision applications where
/// you need to access the raw pixels from camera image
/// on the CPU.
/// 
/// This is different from the ARCameraBackground component, which
/// efficiently displays the camera image on the screen. If you
/// just want to blit the camera texture to the screen, use
/// the ARCameraBackground, or use Graphics.Blit to create
/// a GPU-friendly RenderTexture.
/// 
/// In this example, we get the camera image data on the CPU,
/// convert it to an RGBA format, then display it on the screen
/// as a RawImage texture to demonstrate it is working.
/// This is done as an example; do not use this technique simply
/// to render the camera image on screen.
/// </summary>
public class CameraController : MonoBehaviour
{
    [SerializeField]
    Text m_ImageInfo;

    /// <summary>
    /// The UI Text used to display information about the image on screen.
    /// </summary>
    public Text imageInfo
    {
        get { return m_ImageInfo; }
        set { m_ImageInfo = value; }
    }

    Texture2D m_Texture;
    ARSessionOrigin arOrigin;

    void OnEnable()
    {
        ARSubsystemManager.cameraFrameReceived += OnCameraFrameReceived;

        arOrigin = FindObjectOfType<ARSessionOrigin>();

        InitTF();
        InitIndicator();
    }

    void OnDisable()
    {
        ARSubsystemManager.cameraFrameReceived -= OnCameraFrameReceived;

        CloseTF();
    }

    unsafe void OnCameraFrameReceived(ARCameraFrameEventArgs eventArgs)
    {
        // Attempt to get the latest camera image. If this method succeeds,
        // it acquires a native resource that must be disposed (see below).
        CameraImage image;
        if (!ARSubsystemManager.cameraSubsystem.TryGetLatestImage(out image))
            return;

        // Display some information about the camera image
        m_ImageInfo.text = string.Format(
            "Image info:\n\twidth: {0}\n\theight: {1}\n\tplaneCount: {2}\n\ttimestamp: {3}\n\tformat: {4}",
            image.width, image.height, image.planeCount, image.timestamp, image.format);

        // Choose an RGBA format.
        // See CameraImage.FormatSupported for a complete list of supported formats.
        var format = TextureFormat.RGBA32;

        if (m_Texture == null || m_Texture.width != image.width || m_Texture.height != image.height)
            m_Texture = new Texture2D(image.width, image.height, format, false);

        // Convert the image to format, flipping the image across the Y axis.
        // We can also get a sub rectangle, but we'll get the full image here.
        var conversionParams = new CameraImageConversionParams(image, format, CameraImageTransformation.None);

        // Texture2D allows us write directly to the raw texture data
        // This allows us to do the conversion in-place without making any copies.
        var rawTextureData = m_Texture.GetRawTextureData<byte>();
        try
        {
            image.Convert(conversionParams, new IntPtr(rawTextureData.GetUnsafePtr()), rawTextureData.Length);
        }
        finally
        {
            // We must dispose of the CameraImage after we're finished
            // with it to avoid leaking native resources.
            image.Dispose();
        }

        // Apply the updated texture data to our texture
        m_Texture.Apply();

        // Run TensorFlow inference on the texture
        RunTF(m_Texture);
    }

    [SerializeField]
    TextAsset model;

    [SerializeField]
    TextAsset labels;

    [SerializeField]
    GameObject indicator;

    Classifier classifier;
    Detector detector;

    private IList outputs;
    private GameObject apple;

    public void InitTF()
    {
        // MobileNet
        //classifier = new Classifier(model, labels, output: "MobilenetV1/Predictions/Reshape_1");

        // SSD MobileNet
        detector = new Detector(model, labels, 
                                input: "image_tensor");

        // Tiny YOLOv2
        //detector = new Detector(model, labels, DetectionModels.YOLO,
                                //width: 416,
                                //height: 416,
                                //mean: 0,
                                //std: 255);
    }

    public void InitIndicator()
    {
        apple = Instantiate(indicator, new Vector3(0, 0, 0), Quaternion.identity);
        apple.transform.localScale = new Vector3(0.0004f, 0.0004f, 0.0004f);
        apple.SetActive(false);
    }

    public void RunTF(Texture2D texture)
    {
        // MobileNet
        //outputs = classifier.Classify(texture, angle: 90, threshold: 0.05f);

        // SSD MobileNet
        outputs = detector.Detect(m_Texture, angle: 90, threshold: 0.6f);

        // Tiny YOLOv2
        //outputs = detector.Detect(m_Texture, angle: 90, threshold: 0.1f);

        // Draw AR apple
        for (int i = 0; i < outputs.Count; i++)
        {
            var output = outputs[i] as Dictionary<string, object>;
            if (output["detectedClass"].Equals("apple"))
            {
                DrawApple(output["rect"] as Dictionary<string, float>);
                break;
            }
        }
    }

    public void CloseTF()
    {
        classifier.Close();
        detector.Close();
    }

    public void OnGUI()
    {
        if (outputs != null)
        {
            // Classification
            //Utils.DrawOutput(outputs, new Vector2(20, 20), Color.red);

            // Object detection
            Utils.DrawOutput(outputs, Screen.width, Screen.height, Color.yellow);
        }
    }

    private void DrawApple(Dictionary<string, float> rect)
    {
        var xMin = rect["x"];
        var yMin = 1 - rect["y"];
        var xMax = rect["x"] + rect["w"];
        var yMax = 1 - rect["y"] - rect["h"];

        var pos = GetPosition((xMin + xMax) / 2 * Screen.width, (yMin + yMax) / 2 * Screen.height);
        
        apple.SetActive(true);
        apple.transform.position = pos;
    }

    private Vector3 GetPosition(float x, float y)
    {
        var hits = new List<ARRaycastHit>();

        arOrigin.Raycast(new Vector3(x, y, 0), hits, TrackableType.Planes);

        if (hits.Count > 0)
        {
            var pose = hits[0].pose;

            return pose.position;
        }

        return new Vector3();
    }
}
