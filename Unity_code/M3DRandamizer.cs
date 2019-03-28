using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MORPH3D;
using UnityEngine;

public class M3DRandamizer : MonoBehaviour {

    public GameObject Model;
    public bool isMCSModel;
    public bool isUnityChan;
    public bool isBlueSuitMan;
    public bool isDefaultAvatar;
    public bool isRp_alison_rigged_001;
    public bool isRp_eric_rigged_001_yup_t;
    public bool isBaseball;

    public GameObject[] Jacket;
    public GameObject[] Shirt;
    public GameObject[] Pants;
    public GameObject[] Shose;

    public GameObject[] UnityChanHair;

    

    private List<MORPH3D.COSTUMING.CIclothing> clothingList = null;
    private List<MORPH3D.COSTUMING.CIhair> hairList = null;
    private M3DCharacterManager manager;

    public Material UnityChanBodyMaterial;
    public Texture[] UnityChanBodyTexture;
    public Material UnityChanHairMaterial;
    public Texture[] UnityChanHairTexture;

    public Material BlueSuitManHeadMaterial;
    public Texture[] BlueSuitManHeadTexture;
    public Material BlueSuitManSuitMaterial;
    public Texture[] BlueSuitManSuitTexture;

    public Material DefaultAvatarHeadMaterial;
    public Texture[] DefaultAvatarHeadTexture;
    public Material DefaultAvatarSuitMaterial;
    public Texture[] DefaultAvatarSuitTexture;

    public Material Rp_alison_rigged_001Material;
    public Texture[] Rp_alison_rigged_001Texture;

    public Material Rp_eric_rigged_001_yup_tMaterial;
    public Texture[] Rp_eric_rigged_001_yup_tTexture;

    public Material BaseballCapMaterial;
    public Material BaseballShirtsMaterial;
    public Material BaseballPantsMaterial;

	// GameObject of humanoid joint
    public GameObject rShldrBend;
    public GameObject rForearmBend;
    public GameObject rHand;
    public GameObject rThumb2;
    public GameObject rMid1;

    public GameObject lShldrBend;
    public GameObject lForearmBend;
    public GameObject lHand;
    public GameObject lThumb2;
    public GameObject lMid1;

    public GameObject lEar;
    public GameObject lEye;
    public GameObject rEar;
    public GameObject rEye;
    public GameObject Nose;

    public GameObject rThighBend;
    public GameObject rShin;
    public GameObject rFoot;
    public GameObject rToe;

    public GameObject lThighBend;
    public GameObject lShin;
    public GameObject lFoot;
    public GameObject lToe;

    public GameObject abdomenUpper;

    public float RandomScaleFrom;
    public float RandomScaleTo;

    void Start()
    {
        Init();
    }

    public void MCSMakeModelNaked()
    {
        foreach (MORPH3D.COSTUMING.CIclothing clothing in clothingList)
        {
            manager.SetClothingVisibility(clothing.ID, false);
        }

        foreach (var hair in hairList)
        {
            hair.isVisible = false;
        }
    }

    public void Init()
    {
        if (isMCSModel)
        {
            manager = Model.GetComponent<M3DCharacterManager>();
            clothingList = manager.GetVisibleClothing();
            hairList = manager.GetVisibleHair();
            MCSMakeModelNaked();
        }
    }

    public void Rand()
    {
        if (isMCSModel)
        {
            manager.SetBlendshapeValue("FBMHeavy", Random.Range(0f, 60f));

            MCSMakeModelNaked();

            var jacketIndex = Random.Range(0, Jacket.Length * 2);
            if (jacketIndex < Jacket.Length)
            {
                ShowClothing(Jacket[jacketIndex]);
            }

            var shirtIndex = Random.Range(0, Shirt.Length + 1);
            if (shirtIndex < Shirt.Length)
            {
                ShowClothing(Shirt[shirtIndex]);
            }

            var pantsIndex = Random.Range(0, Pants.Length + 1);
            if (pantsIndex < Pants.Length)
            {
                ShowClothing(Pants[pantsIndex]);
            }

            var shoseIndex = Random.Range(0, Shose.Length / 2 + 1);
            if (shoseIndex < Shose.Length / 2)
            {
                ShowClothing(Shose[shoseIndex * 2]);
                ShowClothing(Shose[shoseIndex * 2 + 1]);
            }

            var hairIndex = Random.Range(0, hairList.Count + 1);
            if (hairIndex < hairList.Count)
            {
                hairList[hairIndex].isVisible = true;
            }
        }

        if (isUnityChan)
        {
            if (Random.Range(0, 2) == 0)
            {
                foreach (var item in UnityChanHair) item.SetActive(true);
            }
            else
            {
                foreach (var item in UnityChanHair) item.SetActive(false);
            }

            var bodyIndex = Random.Range(0, UnityChanBodyTexture.Length);
            if (bodyIndex < UnityChanBodyTexture.Length)
            {
                UnityChanBodyMaterial.SetTexture("_MainTex", UnityChanBodyTexture[bodyIndex]);
            }

            var hairIndex = Random.Range(0, UnityChanHairTexture.Length);
            if (hairIndex < UnityChanHairTexture.Length)
            {
                UnityChanHairMaterial.SetTexture("_MainTex", UnityChanHairTexture[hairIndex]);
            }
        }

        if (isBlueSuitMan)
        {
            var headIndex = Random.Range(0, BlueSuitManHeadTexture.Length);
            if (headIndex < BlueSuitManHeadTexture.Length)
            {
                BlueSuitManHeadMaterial.SetTexture("_MainTex", BlueSuitManHeadTexture[headIndex]);
            }

            var suitIndex = Random.Range(0, BlueSuitManSuitTexture.Length);
            if (suitIndex < BlueSuitManSuitTexture.Length)
            {
                BlueSuitManSuitMaterial.SetTexture("_MainTex", BlueSuitManSuitTexture[suitIndex]);
            }
        }

        if (isDefaultAvatar)
        {
            var headIndex = Random.Range(0, DefaultAvatarHeadTexture.Length);
            if (headIndex < DefaultAvatarHeadTexture.Length)
            {
                DefaultAvatarHeadMaterial.SetTexture("_MainTex", DefaultAvatarHeadTexture[headIndex]);
            }

            var suitIndex = Random.Range(0, DefaultAvatarSuitTexture.Length);
            if (suitIndex < DefaultAvatarSuitTexture.Length)
            {
                DefaultAvatarSuitMaterial.SetTexture("_MainTex", DefaultAvatarSuitTexture[suitIndex]);
            }
        }

        if (isRp_alison_rigged_001)
        {
            var rIndex = Random.Range(0, Rp_alison_rigged_001Texture.Length);
            if (rIndex < Rp_alison_rigged_001Texture.Length)
            {
                Rp_alison_rigged_001Material.SetTexture("_MainTex", Rp_alison_rigged_001Texture[rIndex]);
            }
        }

        if (isRp_eric_rigged_001_yup_t)
        {
            var rIndex = Random.Range(0, Rp_eric_rigged_001_yup_tTexture.Length);
            if (rIndex < Rp_eric_rigged_001_yup_tTexture.Length)
            {
                Rp_eric_rigged_001_yup_tMaterial.SetTexture("_MainTex", Rp_eric_rigged_001_yup_tTexture[rIndex]);
            }
        }

        if(isBaseball)
        {
            BaseballCapMaterial.SetColor("_Color", new Color(Random.Range(0.4f, 1f), Random.Range(0.4f, 1f), Random.Range(0.4f, 1f)));
            BaseballShirtsMaterial.SetColor("_Color", new Color(Random.Range(0.4f, 1f), Random.Range(0.4f, 1f), Random.Range(0.4f, 1f)));
            BaseballPantsMaterial.SetColor("_Color", new Color(Random.Range(0.4f, 1f), Random.Range(0.4f, 1f), Random.Range(0.4f, 1f)));
        }
    }

    private void ShowClothing(GameObject clothing)
    {
        var c = clothingList.Find(x => x.name == clothing.name);

        if(c != null)
        {
            manager.SetClothingVisibility(c.ID, true);
        }

        return;
    }
}
