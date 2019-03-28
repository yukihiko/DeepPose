using MORPH3D;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using UnityEngine;

public class CameraScript : MonoBehaviour {

    public GameObject Background;
    public GameObject[] Foreground;

    public GameObject[] Person;
    /*
    public GameObject M3DMale;
    public GameObject M3DFemale;
    public GameObject UnityChan;
    public GameObject BlueSuitMan;
    public GameObject DefaultAvatar;
    public GameObject rp_alison_rigged_001;
    */
    private GameObject person;
    private int personCounter;

    public Light MainLight;


    public GameObject cube;

    public Canvas DisplayCanvas;
    public Canvas ImageCanvas;

    private float planeDistance;
    private int fileCounter;
    private float timeleft;

    Animator motion;
    public RuntimeAnimatorController[] PersonAnimator;
    private bool isApplyRootMotion;

    bool isKeyDown;
    bool isStart;
    float angle = 180f;
    int loopCounter = 0;
    int startPerson = 0;

    float randomPersonPos = 0.2f;
    float randomPersonAngle = 25f;
    float randomCameraPos = 0.1f;
    float randomCameraAngle = 5f;
    float randomBackgroundaPos = 3f;
    float randomBackgroundAngle = 10f;

    Vector3 personPos;
    Quaternion personRotation;
    Vector3 cameraPos;
    Vector3 backgroundPos;

    //string saveDataPath = "D:/github/DeepPose/data/images/3D_dataset/";
#if UNITY_EDITOR
    const string saveDataPath = "D:/work/3D_dataset/test/";
#else
    const string saveDataPath = "D:/work/3D_dataset/";
#endif
    string datasetPath1 = saveDataPath + "3D_dataset_1/";
    string datasetPath2 = saveDataPath + "3D_dataset_2/";
    string trainListPath1 = saveDataPath + "train3D_1";
    string trainListPath2 = saveDataPath + "train3D_2";
    string testListPath1 = saveDataPath + "test3D_1";
    string testListPath2 = saveDataPath + "test3D_2";
    string BackgroundPath = "D:/work/3D_dataset/" + "Background";

    string datasetPath = "";
    string trainListPath = "";
    string testListPath = "";

    private List<Texture2D> backgroundList = new List<Texture2D>();

    M3DRandamizer randamizer;

    public Camera SnapShotCamera;
    private Texture2D screenShot;
    private CRT noiseCRT;

    // Use this for initialization
    void Start () {

        planeDistance = DisplayCanvas.planeDistance;
        fileCounter = 0;
        timeleft = 0.1f;
        noiseCRT = SnapShotCamera.GetComponent<CRT>();

        foreach (var p in Person)
        {
            p.SetActive(true);
        }

        person = Person[0];
        personCounter = 0;
        RestModel();

        isStart = false;
        isKeyDown = false;
        personPos = person.transform.position;
        personRotation = person.transform.rotation;
        cameraPos = Camera.main.transform.position;
        backgroundPos = Background.transform.position;

        LoadBackground();
        SetBackground();

        screenShot = new Texture2D(SnapShotCamera.targetTexture.width, SnapShotCamera.targetTexture.height, TextureFormat.RGB24, false);

    }

    private void SetPerson(int cnt)
    {
        for (var i = 0; i < Person.Length; i++)
        {
            Person[i].SetActive(i == cnt);
        }
        person = Person[cnt];
        RestModel();

        randamizer = person.GetComponent<M3DRandamizer>();

        motion = person.GetComponent<Animator>();
        if (!randamizer.isBaseball)
        {
#if UNITY_EDITOR
            var animeIndex = Random.Range(0, PersonAnimator.Length);
            animeIndex = 3;
#else
        var animeIndex = Random.Range(0, PersonAnimator.Length);
#endif
            motion.runtimeAnimatorController = PersonAnimator[animeIndex];
            if (animeIndex == 1)
            {
                motion.applyRootMotion = false;
                isApplyRootMotion = false;
            }
            else
            {
                motion.applyRootMotion = true;
                isApplyRootMotion = true;
            }
        }
#if UNITY_EDITOR
        motion.speed = 0.8f;
        startPerson = 16;
#else
        motion.speed = 1f;
        startPerson = 0;
#endif
    }

    void RestModel()
    {
        foreach (var p in Person)
        {
            p.transform.position = new Vector3(0f, 0.0f, 0f);
            p.transform.rotation = Quaternion.AngleAxis(180f, new Vector3(0, 1, 0));
        }
    }

    void Randamize()
    {
        timeleft = 0.2f + Random.Range(-0.05f, 0.05f);
        SetBackground();
        //Camera.main.transform.position = cameraPos + new Vector3(Random.Range(-randomCameraPos, randomCameraPos), Random.Range(-randomCameraPos, randomCameraPos), Random.Range(-randomCameraPos, randomCameraPos));
        //Camera.main.transform.rotation = Quaternion.Euler(Random.Range(-randomCameraAngle, randomCameraAngle), Random.Range(-randomCameraAngle, randomCameraAngle), Random.Range(-randomCameraAngle*3, randomCameraAngle*3));
        Background.transform.position = backgroundPos + new Vector3(Random.Range(-randomBackgroundaPos, randomBackgroundaPos), Random.Range(-randomBackgroundaPos, randomBackgroundaPos), Random.Range(-randomBackgroundaPos, randomBackgroundaPos));
        Background.transform.rotation = Quaternion.Euler(Random.Range(-randomBackgroundAngle, randomBackgroundAngle), Random.Range(-randomBackgroundAngle, randomBackgroundAngle), 180f + Random.Range(-randomBackgroundAngle, randomBackgroundAngle));
        MainLight.color = new Color(Random.Range(0.4f, 1f), Random.Range(0.4f, 1f), Random.Range(0.4f, 1f));
        randamizer.Rand();
        person.transform.position = personPos + new Vector3(Random.Range(-randomPersonPos, randomPersonPos), 0f, Random.Range(-randomPersonPos * 2f, randomPersonPos));
        //person.transform.rotation = Quaternion.AngleAxis(angle + Random.Range(-randomPersonAngle, randomPersonAngle), new Vector3(0, 1, 0));
        person.transform.rotation = personRotation * Quaternion.Euler(Random.Range(-randomCameraAngle, randomCameraAngle), Random.Range(-randomPersonAngle, randomPersonAngle), Random.Range(-randomCameraAngle * 2, randomCameraAngle * 2));

        if (!randamizer.isBaseball)
        {
            var scale = Random.Range(randamizer.RandomScaleFrom, randamizer.RandomScaleTo);
            person.transform.localScale = personPos + new Vector3(scale, scale, scale);
        }
        
        foreach(var f in Foreground)
        {
            f.transform.position = new Vector3(Random.Range(-1.3f, 1.3f), 1f + Random.Range(-1.3f, 1.3f), Random.Range(-0.2f, 0.8f) - 1f);
            f.transform.rotation = Quaternion.Euler(Random.Range(-randomBackgroundAngle, randomBackgroundAngle), Random.Range(-randomBackgroundAngle, randomBackgroundAngle), 180f + Random.Range(-randomBackgroundAngle, randomBackgroundAngle));
            var scale = Random.Range(0.2f, 0.3f);
            f.transform.localScale = personPos + new Vector3(scale, scale, 0.0001f);
        }
        
        /*
        if (Random.Range(0, 2) == 0)
        {
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.Exponential;
            RenderSettings.fogDensity = Random.Range(0.0f, 0.015f);
        }
        */
        noiseCRT.RGBNoise = Random.Range(0.0f, 0.1f);
    }

    private void StartAnime(int pathCnt)
    {
        personCounter = startPerson;
        SetPerson(personCounter);

        if (pathCnt == 2)
        {
            datasetPath = datasetPath2;
            trainListPath = trainListPath2;
            testListPath = testListPath2;
        }
        else
        {
            datasetPath = datasetPath1;
            trainListPath = trainListPath1;
            testListPath = testListPath1;
        }

        motion.SetBool("IsStart", true);
        isKeyDown = true;
        fileCounter = 0;
        timeleft = 0f;
        loopCounter = 0;

    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown("space"))
        {
            //motion.SetBool("IsStart", true);
            personCounter = 0;
            SetPerson(personCounter);
            isKeyDown = true;
            fileCounter = 0;
            loopCounter = 0;
            Randamize();
            return;
        }
        
        if(!isKeyDown) return;

        if (datasetPath == "")
        {
            if (System.IO.File.Exists(trainListPath1))
            {
                if (System.IO.File.Exists(trainListPath2))
                {
                    // Waiting
                }
                else
                {
                    StartAnime(2);
                }
            }
            else
            {
                StartAnime(1);
            }
        }

        if (datasetPath == "")
        {
            return;
        }

        if (isStart && motion.GetCurrentAnimatorStateInfo(0).IsName("Start"))
        {
            loopCounter++;
            if (loopCounter == 4)
            {
                // Finish
                RestModel();
                isStart = false;
                motion.applyRootMotion = true;
                motion.SetBool("IsStart", false);
                loopCounter = 0;

                if (personCounter == Person.Length - 1)
                {
                    // Finish
                    datasetPath = "";
                    trainListPath = "";
                    testListPath = "";
                    return;
                }
                else
                {
                    // Change Model
                    personCounter++;
                    SetPerson(personCounter);
                    motion.SetBool("IsStart", true);
                }
            }
            angle += 90f;
            person.transform.position = new Vector3(0f, 0f, 0f);
            person.transform.rotation = Quaternion.AngleAxis(angle, new Vector3(0, 1, 0));
            personRotation = person.transform.rotation;
            isStart = false;
            motion.applyRootMotion = isApplyRootMotion;
            return;
        }

        if (!motion.GetCurrentAnimatorStateInfo(0).IsName("Start"))
        {
            isStart = true;
        }

        if (motion.applyRootMotion && motion.GetCurrentAnimatorStateInfo(0).IsName("Breakdance"))
        {
            motion.applyRootMotion = false;
        }

        timeleft -= Time.deltaTime;
        if (timeleft <= 0.0)
        {
            DisplayCanvas.planeDistance = GetPlaneDistance();
            SnapShotCamera.Render();

            var wLetTop = Vector3.zero;
            var wLeftBottom = Vector3.zero;
            //var rightTop = Vector3.zero;
            //var rightBottom = Vector3.zero;
            var uiCamera = SnapShotCamera;
            var canvasRect = DisplayCanvas.GetComponent<RectTransform>();
            var w = canvasRect.rect.width / 2f;
            var h = canvasRect.rect.height / 2f;
            RectTransformUtility.ScreenPointToWorldPointInRectangle(canvasRect, new Vector2(0, h + w), uiCamera, out wLetTop);
            RectTransformUtility.ScreenPointToWorldPointInRectangle(canvasRect, new Vector2(0, h - w), uiCamera, out wLeftBottom);
            var dist = Vector3.Distance(wLetTop, wLeftBottom);

            var letTop = SnapShotCamera.transform.InverseTransformPoint(wLetTop);
            var pos_abdomenUpper = (GetP(randamizer.abdomenUpper) - letTop) / dist;
            var pos_rShldrBend = (GetP(randamizer.rShldrBend) - letTop) / dist;
            var pos_rForearmBend = (GetP(randamizer.rForearmBend) - letTop) / dist;
            var pos_rHand = (GetP(randamizer.rHand) - letTop) / dist;
            var pos_rThumb2 = (GetP(randamizer.rThumb2) - letTop) / dist;
            var pos_rMid1 = (GetP(randamizer.rMid1) - letTop) / dist;
            var pos_lShldrBend = (GetP(randamizer.lShldrBend) - letTop) / dist;
            var pos_lForearmBend = (GetP(randamizer.lForearmBend) - letTop) / dist;
            var pos_lHand = (GetP(randamizer.lHand) - letTop) / dist;
            var pos_lThumb2 = (GetP(randamizer.lThumb2) - letTop) / dist;
            var pos_lMid1 = (GetP(randamizer.lMid1) - letTop) / dist;
            var pos_lEar = (GetP(randamizer.lEar) - letTop) / dist;
            var pos_lEye = (GetP(randamizer.lEye) - letTop) / dist;
            var pos_rEar = (GetP(randamizer.rEar) - letTop) / dist;
            var pos_rEye = (GetP(randamizer.rEye) - letTop) / dist;
            var pos_Nose = (GetP(randamizer.Nose) - letTop) / dist;
            var pos_rThighBend = (GetP(randamizer.rThighBend) - letTop) / dist;
            var pos_rShin = (GetP(randamizer.rShin) - letTop) / dist;
            var pos_rFoot = (GetP(randamizer.rFoot) - letTop) / dist;
            var pos_rToe = (GetP(randamizer.rToe) - letTop) / dist;
            var pos_lThighBend = (GetP(randamizer.lThighBend) - letTop) / dist;
            var pos_lShin = (GetP(randamizer.lShin) - letTop) / dist;
            var pos_lFoot = (GetP(randamizer.lFoot) - letTop) / dist;
            var pos_lToe = (GetP(randamizer.lToe) - letTop) / dist;

            var pos2D_abdomenUpper =GetP2D(randamizer.abdomenUpper);
            var pos2D_rShldrBend = GetP2D(randamizer.rShldrBend);
            var pos2D_rForearmBend = GetP2D(randamizer.rForearmBend);
            var pos2D_rHand = GetP2D(randamizer.rHand);
            var pos2D_rThumb2 = GetP2D(randamizer.rThumb2);
            var pos2D_rMid1 = GetP2D(randamizer.rMid1);
            var pos2D_lShldrBend = GetP2D(randamizer.lShldrBend);
            var pos2D_lForearmBend = GetP2D(randamizer.lForearmBend);
            var pos2D_lHand = GetP2D(randamizer.lHand);
            var pos2D_lThumb2 = GetP2D(randamizer.lThumb2);
            var pos2D_lMid1 = GetP2D(randamizer.lMid1);
            var pos2D_lEar = GetP2D(randamizer.lEar);
            var pos2D_lEye = GetP2D(randamizer.lEye);
            var pos2D_rEar = GetP2D(randamizer.rEar);
            var pos2D_rEye = GetP2D(randamizer.rEye);
            var pos2D_Nose = GetP2D(randamizer.Nose);
            var pos2D_rThighBend = GetP2D(randamizer.rThighBend);
            var pos2D_rShin = GetP2D(randamizer.rShin);
            var pos2D_rFoot = GetP2D(randamizer.rFoot);
            var pos2D_rToe = GetP2D(randamizer.rToe);
            var pos2D_lThighBend = GetP2D(randamizer.lThighBend);
            var pos2D_lShin = GetP2D(randamizer.lShin);
            var pos2D_lFoot = GetP2D(randamizer.lFoot);
            var pos2D_lToe = GetP2D(randamizer.lToe);

            cube.transform.position = Camera.main.transform.TransformPoint(pos_lToe* dist + letTop);

            fileCounter++;
            var fileName = fileCounter.ToString("00000");

            string imgOrg = saveDataPath + "org/im" + fileName + ".png";
            string imgPath = datasetPath + "im" + fileName + ".png";
            if (File.Exists(imgPath))
            {
                //File.Delete(imgOrg);
                //File.Delete(imgPath);
            }
            //ScreenCapture.CaptureScreenshot(imgOrg);
            SaveCameraImage(imgPath);

            var str = datasetPath + "im" + fileName + ".png," + (DisplayCanvas.planeDistance/dist).ToString() + "," +
                ToStr(pos_rShldrBend, pos2D_rShldrBend) + ToStr(pos_rForearmBend, pos2D_rForearmBend) + ToStr(pos_rHand, pos2D_rHand) + ToStr(pos_rThumb2, pos2D_rThumb2) + ToStr(pos_rMid1, pos2D_rMid1) +
                ToStr(pos_lShldrBend, pos2D_lShldrBend) + ToStr(pos_lForearmBend, pos2D_lForearmBend) + ToStr(pos_lHand, pos2D_lHand) + ToStr(pos_lThumb2, pos2D_lThumb2) + ToStr(pos_lMid1, pos2D_lMid1) +
                ToStr(pos_lEar, pos2D_lEar) + ToStr(pos_lEye, pos2D_lEye) + ToStr(pos_rEar, pos2D_rEar) + ToStr(pos_rEye, pos2D_rEye) + ToStr(pos_Nose, pos2D_Nose) +
                ToStr(pos_rThighBend, pos2D_rThighBend) + ToStr(pos_rShin, pos2D_rShin) + ToStr(pos_rFoot, pos2D_rFoot) + ToStr(pos_rToe, pos2D_rToe) +
                ToStr(pos_lThighBend, pos2D_lThighBend) + ToStr(pos_lShin, pos2D_lShin) + ToStr(pos_lFoot, pos2D_lFoot) + ToStr(pos_lToe, pos2D_lToe) + ToStr(pos_abdomenUpper, pos2D_abdomenUpper, true);
            /*
            ToStr(pos_rShldrBend)0 + ToStr(pos_rForearmBend)1 + ToStr(pos_rHand)2 + ToStr(pos_rThumb2)3 + ToStr(pos_rMid1)4 +
            ToStr(pos_lShldrBend)5 + ToStr(pos_lForearmBend)6 + ToStr(pos_lHand) 7+ ToStr(pos_lThumb2)8 + ToStr(pos_lMid1)9 +
            ToStr(pos_lEar)10 + ToStr(pos_lEye)11 + ToStr(pos_rEar)12 + ToStr(pos_rEye)13 + ToStr(pos_Nose)14 +
            ToStr(pos_rThighBend)15 + ToStr(pos_rShin)16 + ToStr(pos_rFoot)17 + ToStr(pos_rToe)18 +
            ToStr(pos_lThighBend)19 + ToStr(pos_lShin)20 + ToStr(pos_lFoot)21 + ToStr(pos_lToe)22 + ToStr(pos_abdomenUpper, true)23;
            [0,1],[1,2],[2,3],[2,4],
            [5,6],[6,7],[7,8],[7,9],
            [10,11],[11,14],[14,13],[13,12],
            [15,16],[16,17],[17,18],
            [19,20],[20,21],[21,22],
            [10,0],[12,5],
            [0,23],[5,23],
            [15,23],[19,23],
            [0,15],[5,19],
        */
            if (trainListPath == null || trainListPath =="")
            {
                trainListPath = "";
            }
            if (Random.Range(0, 20) == 10)
            {
                SaveTestCsv(str);
            }
            else
            {
                SaveTranCsv(str);
            }

            Randamize();
/*            
            var exProcess = new Process();

#if UNITY_EDITOR
            exProcess.StartInfo.FileName = Application.dataPath + "/Data/ImgResizeCmd.exe";
#else
            exProcess.StartInfo.FileName  = saveDataPath + "bin/ImgResizeCmd.exe";
#endif
            exProcess.StartInfo.Arguments = imgOrg + " " + imgPath;
            exProcess.StartInfo.CreateNoWindow = true;
            exProcess.StartInfo.UseShellExecute = false;

            //外部プロセスの終了を検知してイベントを発生させます.
            //exProcess.EnableRaisingEvents = true;
            //exProcess.Exited += exProcess_Exited;

            //外部のプロセスを実行する
            exProcess.Start();
*/
        }
    }

    // Update is called once per frame
    float GetPlaneDistance()
    {
        return GetZ(randamizer.abdomenUpper);

        /*
        return (GetZ(rShldrBend) + GetZ(rForearmBend) + GetZ(rHand) + GetZ(rThumb2) + GetZ(rMid1) +
            GetZ(lShldrBend) + GetZ(lForearmBend) + GetZ(lHand) + GetZ(lThumb2) + GetZ(lMid1) +
            GetZ(lEar) + GetZ(lEye) + GetZ(rEar) + GetZ(rEye) + GetZ(Nose) +
            GetZ(rThighBend) + GetZ(rShin) + GetZ(rFoot) + GetZ(rToe) +
            GetZ(lThighBend) + GetZ(lShin) + GetZ(lFoot) + GetZ(lToe)) + GetZ(abdomenUpper) / 24.0f;
            */
            /*
            return (rShldrBend.transform.position.z + rForearmBend.transform.position.z + rHand.transform.position.z + rThumb2.transform.position.z + rMid1.transform.position.z +
            lShldrBend.transform.position.z + lForearmBend.transform.position.z + lHand.transform.position.z + lThumb2.transform.position.z + lMid1.transform.position.z +
            lEar.transform.position.z + lEye.transform.position.z + rEar.transform.position.z + rEye.transform.position.z + Nose.transform.position.z +
            rThighBend.transform.position.z + rShin.transform.position.z + rFoot.transform.position.z + rToe.transform.position.z +
            lThighBend.transform.position.z + lShin.transform.position.z + lFoot.transform.position.z + lToe.transform.position.z) / 23.0f;
            */
        }

    Vector3 GetP(GameObject obj)
    {
        return SnapShotCamera.transform.InverseTransformPoint(obj.transform.position);
    }

    Vector3 GetP2D(GameObject obj)
    {
        return RectTransformUtility.WorldToScreenPoint(SnapShotCamera, obj.transform.position)/224f;
    }

    float GetZ(GameObject obj)
    {
        return SnapShotCamera.transform.InverseTransformPoint(obj.transform.position).z;
    }

    string ToStr(Vector3 pos, Vector2 pos2D, bool endF = false)
    {
        string vis = ",1.0";
        if ((pos2D.x < 0f || pos2D.x > 224f) || (pos2D.y < 0f || pos2D.y > 224f))
        {
            vis = ",0.0";
        }
        return pos.x.ToString("0.#########") + "," + (pos.y * -1.0).ToString("0.#########") + "," + pos.z.ToString("0.#########") + "," + pos2D.x.ToString("0.#########") + "," + (1f - pos2D.y).ToString("0.#########") + vis + (!endF ? "," : "");
    }

    public void SaveTranCsv(string txt)
    {
        SaveCsv(trainListPath, txt);
    }
    public void SaveTestCsv(string txt)
    {
        SaveCsv(testListPath, txt);
    }

    public void SaveCsv(string fileName, string txt)
    {
        StreamWriter sw;
        FileInfo fi;
        fi = new FileInfo(fileName);
        sw = fi.AppendText();
        sw.WriteLine(txt);
        sw.Flush();
        sw.Close();
    }

    public void SaveCameraImage(string imgOrg)
    {
        // Remember currently active render textureture
        RenderTexture currentRT = RenderTexture.active;
        // Set the supplied RenderTexture as the active one
        RenderTexture.active = SnapShotCamera.targetTexture;
        Camera.main.Render();
        SnapShotCamera.Render();
        // Create a new Texture2D and read the RenderTexture texture into it
        screenShot.ReadPixels(new Rect(0, 0, SnapShotCamera.targetTexture.width, SnapShotCamera.targetTexture.height), 0, 0);
        //転送処理の適用
        screenShot.Apply();
        // Restorie previously active render texture to avoid errors
        RenderTexture.active = currentRT;
        //PNGに変換
        byte[] bytes = screenShot.EncodeToPNG();
        //保存
        File.WriteAllBytes(imgOrg, bytes);
    }

    void LoadBackground()
    {
        backgroundList.Clear();
        string[] files = System.IO.Directory.GetFiles(BackgroundPath, "*.png", System.IO.SearchOption.AllDirectories);
        foreach (var f in files)
        {
            backgroundList.Add(PngToTex2D(f));
        }
    }

    void SetBackground()
    {
        var i = Random.Range(0, backgroundList.Count);
        Background.GetComponent<Renderer>().material.mainTexture = backgroundList[i];
        foreach (var f in Foreground)
        {
            f.GetComponent<Renderer>().material.mainTexture = backgroundList[Random.Range(0, backgroundList.Count)];
        }
    }

    Texture2D PngToTex2D(string path)
    {
        BinaryReader bin = new BinaryReader(new FileStream(path, FileMode.Open, FileAccess.Read));
        byte[] rb = bin.ReadBytes((int)bin.BaseStream.Length);
        bin.Close();
        int pos = 16, width = 0, height = 0;
        for (int i = 0; i < 4; i++) width = width * 256 + rb[pos++];
        for (int i = 0; i < 4; i++) height = height * 256 + rb[pos++];
        Texture2D texture = new Texture2D(width, height);
        texture.LoadImage(rb);
        return texture;
    }
}
