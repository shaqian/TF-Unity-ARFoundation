using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum Flip {
    NONE,
    VERTICAL,
    HORIZONTAL
}

public static class Utils
{
    private static Color32[] rotateSquare(Color32[] arr, double phi, Texture2D tex)
    {
        int x;
        int y;
        int i;
        int j;
        double sn = Math.Sin(phi);
        double cs = Math.Cos(phi);
        Color32[] arr2 = tex.GetPixels32();
        int W = tex.width;
        int H = tex.height;
        int xc = W / 2;
        int yc = H / 2;
        for (j = 0; j < H; j++)
        {
            for (i = 0; i < W; i++)
            {
                arr2[j * W + i] = new Color32(0, 0, 0, 0);
                x = (int)(cs * (i - xc) + sn * (j - yc) + xc);
                y = (int)(-sn * (i - xc) + cs * (j - yc) + yc);
                if ((x > -1) && (x < W) && (y > -1) && (y < H))
                {
                    arr2[j * W + i] = arr[y * W + x];
                }
            }
        }
        return arr2;
    }

    public static Color32[] RotateImage(Texture2D tex, int angle)
    {
        Color32[] pix1 = tex.GetPixels32();
        int W = tex.width;
        int H = tex.height;
        int x = 0;
        int y = 0;
        Color32[] pix2 = rotateSquare(pix1, (Math.PI / 180 * (double)angle), tex);
        for (int j = 0; j < H; j++)
        {
            for (var i = 0; i < W; i++)
            {
                pix1[x + i + W * (j + y)] = pix2[i + j * W];
            }
        }
        //tex.SetPixels32(pix1);
        //tex.Apply();

        //return tex.GetPixels32();

        return pix1;
    }

    public static Color32[] GetPixels(Texture2D tex, int width, int height, int angle, Flip flip)
    {
        Rect texR = new Rect(0, 0, width, height);

        tex.filterMode = FilterMode.Trilinear;
        tex.Apply(true);

        RenderTexture rtt = new RenderTexture(width, height, 32);
        Graphics.SetRenderTarget(rtt);

        switch(flip)
        {
            case Flip.VERTICAL:
                GL.LoadPixelMatrix(0, 1, 0, 1);
                break;
            case Flip.HORIZONTAL:
                GL.LoadPixelMatrix(1, 0, 1, 0);
                break;
            default:
                GL.LoadPixelMatrix(0, 1, 1, 0);
                break;
        }

        GL.Clear(true, true, new Color(0, 0, 0, 0));
        Graphics.DrawTexture(new Rect(0, 0, 1, 1), tex);

        tex.Resize(width, height);
        tex.ReadPixels(texR, 0, 0, true);
        tex.Apply(true);

        var pixels = RotateImage(tex, angle);

        UnityEngine.Object.Destroy(tex);
        tex = null;
        UnityEngine.Object.Destroy(rtt);
        rtt = null;

        return pixels;
    }

    public static byte[] DecodeTexture(Texture2D texture, int width, int height, int angle, Flip flip)
    {
        var pixels = GetPixels(texture, width, height, angle, flip);

        int pixel = 0;
        int index = 0;

        byte[] imgData = new byte[height * width * 3];

        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                var color = pixels[pixel++];
                imgData[index++] = color.r;
                imgData[index++] = color.g;
                imgData[index++] = color.b;
            }
        }

        return imgData;
    }

    public static float[] DecodeTexture(Texture2D texture, int width, int height, 
                                        float mean, float std, int angle, Flip flip)
    {
        var pixels = GetPixels(texture, width, height, angle, flip);

        int pixel = 0;
        int index = 0;

        float[] imgData = new float[height * width * 3];

        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                var color = pixels[pixel++];
                imgData[index++] = (color.r - mean) / std;
                imgData[index++] = (color.g - mean) / std;
                imgData[index++] = (color.b - mean) / std;
            }
        }

        return imgData;
    }

    public static void Log(List<Dictionary<string, object>> results)
    {
        results.ForEach(result =>
        {
            var confidence = result["confidenceInClass"];
            var detectedClass = result["detectedClass"];
            var rect = (Dictionary<string, float>)result["rect"];
            Debug.Log($"{detectedClass} - {confidence} - {rect["x"]} - {rect["y"]} - {rect["w"]} - {rect["h"]}");
        });
    }

    public static void Log(List<KeyValuePair<string, float>> results)
    {
        results.ForEach(result =>
        {
            Debug.Log($"{result.Key} - {result.Value}");
        });
    }

    static List<KeyValuePair<Color, Texture2D>> textures = new List<KeyValuePair<Color, Texture2D>>();

    public static void DrawRect(Rect area, int frameWidth, Color color)
    {
        var tmp = textures.Find(t => t.Key == color);
        Texture2D texture;
        if (tmp.Equals(default(KeyValuePair<Color, Texture2D>)))
        {
            texture = new Texture2D(1, 1);
            texture.SetPixel(0, 0, color);
            texture.Apply();
            textures.Add(new KeyValuePair<Color, Texture2D>(color, texture));
        }
        else
        {
            texture = tmp.Value;
        }

        Rect lineArea = area;
        lineArea.height = frameWidth; //Top line
        GUI.DrawTexture(lineArea, texture);
        lineArea.y = area.yMax - frameWidth; //Bottom
        GUI.DrawTexture(lineArea, texture);
        lineArea = area;
        lineArea.width = frameWidth; //Left
        GUI.DrawTexture(lineArea, texture);
        lineArea.x = area.xMax - frameWidth;//Right
        GUI.DrawTexture(lineArea, texture);
    }

    public static void DrawText(Rect area, string text, GUIStyle style)
    {
        GUI.Label(area, text, style);
    }

    public static void DrawOutput(IList outputs, int width, int height, Color color)
    {
        var list = outputs as List<Dictionary<string, object>>;
        list.ForEach(output =>
        {
            var rect = output["rect"] as Dictionary<string, float>;

            Utils.DrawRect(
                new Rect(
                    rect["x"] * width,
                    rect["y"] * height,
                    rect["w"] * width,
                    rect["h"] * height),
                5,
                color);

            var style = new GUIStyle();
            style.fontSize = 50;
            style.normal.textColor = color;

            Utils.DrawText(
                new Rect(
                    rect["x"] * width + 5,
                    rect["y"] * height + 5,
                    0,
                    0),
                $"{output["detectedClass"]} - {output["confidenceInClass"]}",
                style);
        });
    }

    public static void DrawOutput(IList outputs, Vector2 position, Color color)
    {
        var list = outputs as List<KeyValuePair<string, float>>;
        var cnt = 0;

        list.ForEach(output =>
        {
            var style = new GUIStyle();
            style.fontSize = 50;
            style.normal.textColor = color;

            Utils.DrawText(
                new Rect(position.x, position.y + cnt * 70, 0, 0), 
                $"{output.Key} - {output.Value}", 
                style);

            cnt++;
        });
    }
}
