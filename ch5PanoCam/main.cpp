#include <opencv2/opencv.hpp> 
#include <thread>        
#include <mutex> 
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/xphoto.hpp"
#endif

using namespace cv;
using namespace std;
using namespace cv::detail;

#define NBCAMERA 10
#define MODE_AFFICHAGE 0x100

vector<mutex> mtxTimeStamp(NBCAMERA);
int stopThread = 0;
double delai = 0.025;

struct ParamCamera {
    VideoCapture *v;
    int64 tpsCapture;
    int index;
    bool captureImage;
    int cmd;
    Mat imAcq;
    int64 debAcq, finAcq;
    Mat derniereImage;
};

struct ParamPano {
    bool init = false;
    float seuilConfiance = 0.3f;
    float matchConf = 0.25f;
    vector<int> indices;
    string fctCout = "ray";
    string typeCouture = "dp_color";
    string surfaceCompo = "plane";
    int correctionExposition = ExposureCompensator::GAIN_BLOCKS;
    float forceMelange = 100;
    bool coutureVisible = false;
    vector<Size> tailleImages;
    vector<Point> posCoin;
    vector<Size> tailleMasque;
    vector<UMat> masqueCompo;
    vector<UMat> masqueCompoPiece;
    vector<UMat> masquePiece;
    vector<Mat> gains;
    vector<CameraParams> cameras;
    vector<double> focales;
    float focaleMoyenne;
    Ptr<WarperCreator> algoComposition;
    Ptr<ExposureCompensator> algoCorrectExpo;
    Ptr<RotationWarper> composition;
};

static vector<VideoCapture> RechercheCamera();
static void AcquisitionVideo(ParamCamera *pc);
static void GestionCmdCamera(ParamCamera *pc);
static vector<Mat> LireImages(vector<ParamCamera> *pc);
static vector<Mat> LireImagesSynchro(vector<ParamCamera> *pc);
static Mat ComposerPanorama(vector<Mat> matUSB, ParamPano &pp);
static int InitPanorama(vector<Mat> matUSB, ParamPano &pp);
static bool ChargerParamPano(ParamPano &pp);
static void SauverParamPano(ParamPano pp);

int main(int argc, char **argv)
{
    ParamPano pPano;
    ParamPano pPano0;
    int zoom = 16;

    bool panoActif = ChargerParamPano(pPano);
    vector<VideoCapture> v;

    v = RechercheCamera();
    Size tailleGlobale(0, 0);
    vector<ParamCamera> pCamera(v.size());
    if (pPano.cameras.size() != v.size())
    {
        cout << "Nombre de caméra du fichier incohérent\n";
        panoActif = false;
    }
    for (int i = 0; i<v.size(); i++)
    {
        if (v[i].isOpened())
        {
            if (panoActif)
            {
                //               v[i].set(CAP_PROP_FRAME_WIDTH,pPano.tailleImages[i].width );
                //               v[i].set(CAP_PROP_FRAME_HEIGHT,pPano.tailleImages[i].height);
            }
            Size tailleWebcam = Size(static_cast<int>(v[i].get(CAP_PROP_FRAME_WIDTH)), static_cast<int>(v[i].get(CAP_PROP_FRAME_HEIGHT)));
            if (i == 0)
                tailleGlobale = tailleWebcam;
            else if (i % 2 == 1)
                tailleGlobale.width += tailleWebcam.width;
            else
                tailleGlobale.height += tailleWebcam.height;
            pCamera[i].v = &v[i];
            pCamera[i].cmd = 0;
            pCamera[i].index = i;
            thread t(AcquisitionVideo, &pCamera[i]);
            t.detach();
        }
    }
    if (pPano.cameras.size()>v.size())
        panoActif = false;
    else if (panoActif)
        panoActif = pPano.init;
    Mat frame = Mat::zeros(tailleGlobale.height, tailleGlobale.width, CV_8UC3);
    imshow("Pano", frame);
    Mat rPano;
    int modeAffichage = MODE_AFFICHAGE;
    int cameraSelect = 0;
    vector<Mat> x;
    bool cameraNonPrete = true;
    int nbEssai = 0;
    do
    {
        x = LireImages(&pCamera);
        nbEssai++;
        if (x.size() == v.size())
        {
            cameraNonPrete = false;
            for (int i = 0; i<x.size(); i++)
                if (x[i].size().area() == 0)
                    cameraNonPrete = true;
        }
        else
            cameraNonPrete = true;
    } while (cameraNonPrete && nbEssai<100);

    int code = 0;
    while (code != 27)
    {
        code = waitKey(1);
        if (code != -1)
            cout << code << endl;
        switch (code) {
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            if (code - 48 >= 0 && code - 48<v.size())
                cameraSelect = code - 48;
            break;
        case 'g':
        case 'G':
        case 'b':
        case 'B':
        case 'E':
        case 'e':
        case 'c':
        case 'C':
        case 'w':
            mtxTimeStamp[cameraSelect].lock();
            pCamera[cameraSelect].cmd = pCamera[cameraSelect].cmd | code;
            mtxTimeStamp[cameraSelect].unlock();
            break;
        case '+':
            if (zoom<128)
                zoom *= 2;
            break;
        case '-':
            if (zoom>2)
                zoom /= 2;
            break;
        case 's':
            pPano0.surfaceCompo = "stereographic";
        case 'S':
            if (code == 'S')
                pPano0.surfaceCompo = "spherical";
        case 'f':
            if (code == 'f')
                pPano0.surfaceCompo = "fisheye";
        case 'p':
            if (code == 'p')
                pPano0.surfaceCompo = "plane";
        case 'y':
            if (code == 'y')
                pPano0.surfaceCompo = "cylindrical";
            panoActif = false;
            pPano = pPano0;
            x = LireImagesSynchro(&pCamera);
            if (InitPanorama(x, pPano) == 0)
                panoActif = true;
            else
            {
                frame = Mat::zeros(tailleGlobale.height, tailleGlobale.width, CV_8UC3);
                panoActif = false;
            }
            break;
        case 'a':
            modeAffichage = !modeAffichage;
            break;
        case 'r':
            panoActif = false;
            delai = 0.025;
            pPano = pPano0;
            frame = Mat::zeros(tailleGlobale.height, tailleGlobale.width, CV_8UC3);
            break;
        case 'l':
            pPano.coutureVisible = !pPano.coutureVisible;
            pPano.masquePiece.clear();
            pPano.masquePiece.resize(pPano.cameras.size());
            break;
        }
        if (panoActif)
        {
            x = LireImagesSynchro(&pCamera);
            rPano = ComposerPanorama(x, pPano);
            Rect r(0, 0, rPano.cols, rPano.rows);
        }
        else
        {
            x = LireImages(&pCamera);
            if (x.size() == v.size())
                for (int j = 0; j<v.size(); j++)
                {
                    if (v[j].isOpened())
                    {
                        int k = j;
                        if (j == 0)
                            k = 0;
                        else
                            k = j - 1;
                        Rect r(k % 2 * x[k].cols, k / 2 * x[k].rows, x[j].cols, x[j].rows);
                        x[j].copyTo(frame(r));
                    }
                }
        }
        if (modeAffichage && x.size() == v.size())
        {
            for (int i = 0; i < v.size(); i++)
            {
                if (!pCamera[i].derniereImage.empty())
                    if (zoom == 16)
                        imshow(format("Webcam %d", i), x[i]);
                    else
                    {
                        Mat tmp;
                        resize(x[i], tmp, Size(), zoom / 16.0, zoom / 16.0);
                        imshow(format("Webcam %d", i), tmp);
                    }
            }
        }
        Mat frame2;
        frame2 = frame;
        if (panoActif)
        {
            if (zoom == 16)
                imshow("Pano", rPano);
            else
            {
                Mat tmp;
                resize(rPano, tmp, Size(), zoom / 16.0, zoom / 16.0);
                imshow("Pano", tmp);
            }
        }
        else if (zoom == 16)
            imshow("Pano", frame2);
        else
        {
            Mat tmp;
            resize(frame2, tmp, Size(), zoom / 16.0, zoom / 16.0);
            imshow("Pano", tmp);
        }
    }
    for (int j = 0; j<v.size(); j++)
        if (v[static_cast<int>(j)].isOpened())
            mtxTimeStamp[j].lock();
    stopThread = 1;
    for (int j = 0; j<v.size(); j++)
        if (v[static_cast<int>(j)].isOpened())
            mtxTimeStamp[j].unlock();
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return 0;
}

int InitPanorama(vector<Mat> matUSB, ParamPano &pp)
{
    int nbImages = static_cast<int>(matUSB.size());

    if (nbImages < 2)
    {
        cout << "Il faut au moins 2 images\n";
        return 1;
    }
    Ptr<Feature2D> rechercheDescripteur;
    vector<ImageFeatures> descripteurs(nbImages);

#ifdef HAVE_OPENCV_XFEATURES2D
    rechercheDescripteur = cv::xfeatures2d::SURF::create();
#else
    rechercheDescripteur = makePtr<OrbFeaturesFinder>();
#endif
    for (int i = 0; i < nbImages; ++i)
    {
        computeImageFeatures(rechercheDescripteur,matUSB[i], descripteurs[i]);
        descripteurs[i].img_idx = i;
    }
    vector<MatchesInfo> appariementImage;
    BestOf2NearestMatcher apparier(false, pp.matchConf);
    apparier(descripteurs, appariementImage);
    pp.indices = leaveBiggestComponent(descripteurs, appariementImage, pp.seuilConfiance);
    nbImages = static_cast<int>(pp.indices.size());
    if (nbImages < 2)
        return 2;
    pp.tailleImages.resize(nbImages);
    pp.tailleMasque.resize(nbImages);
    pp.posCoin.resize(nbImages);
    pp.masqueCompo.resize(nbImages);
    pp.masqueCompoPiece.resize(nbImages);
    pp.masquePiece.resize(nbImages);
    for (int i = 0; i < nbImages; ++i)
    {
        pp.tailleImages[i] = matUSB[pp.indices[i]].size();
    }
    HomographyBasedEstimator estimHomographie;
    if (!estimHomographie(descripteurs, appariementImage, pp.cameras))
    {
        cout << "Echec de l'estimation de l'homographie.\n";
        return 3;
    }
    for (size_t i = 0; i < pp.cameras.size(); ++i)
    {
        Mat R;
        pp.cameras[i].R.convertTo(R, CV_32F);
        pp.cameras[i].R = R;
    }
    Ptr<detail::BundleAdjusterBase> ajuster;
    if (pp.fctCout == "reproj")
        ajuster = makePtr<detail::BundleAdjusterReproj>();
    else if (pp.fctCout == "ray")
        ajuster = makePtr<detail::BundleAdjusterRay>();
    else
    {
        cout << "Unknown bundle adjustment cost function: '" << pp.fctCout << "'.\n";
        return 4;
    }
    ajuster->setConfThresh(pp.seuilConfiance);
    Mat_<uchar> refine_mask = Mat::ones(3, 3, CV_8U);
    ajuster->setRefinementMask(refine_mask);
    if (!(*ajuster)(descripteurs, appariementImage, pp.cameras))
    {
        cout << "Echec de l'ajustement des parametres.\n";
        return 5;
    }
    for (size_t i = 0; i < pp.cameras.size(); ++i)
        pp.focales.push_back(pp.cameras[i].focal);
    sort(pp.focales.begin(), pp.focales.end());
    if (pp.focales.size() % 2 == 1)
        pp.focaleMoyenne = static_cast<float>(pp.focales[pp.focales.size() / 2]);
    else
        pp.focaleMoyenne = static_cast<float>(pp.focales[pp.focales.size() / 2 - 1] + pp.focales[pp.focales.size() / 2]) * 0.5f;

    vector<UMat> imagesProjetees(nbImages);
    vector<UMat> masques(nbImages);

    for (int i = 0; i < nbImages; ++i)
    {
        masques[i].create(matUSB[pp.indices[i]].size(), CV_8U);
        masques[i].setTo(Scalar::all(255));
    }
    if (pp.surfaceCompo == "plane")
        pp.algoComposition = makePtr<cv::PlaneWarper>();
    else if (pp.surfaceCompo == "cylindrical")
        pp.algoComposition = makePtr<cv::CylindricalWarper>();
    else if (pp.surfaceCompo == "spherical")
        pp.algoComposition = makePtr<cv::SphericalWarper>();
    else if (pp.surfaceCompo == "fisheye")
        pp.algoComposition = makePtr<cv::FisheyeWarper>();
    else if (pp.surfaceCompo == "stereographic")
        pp.algoComposition = makePtr<cv::StereographicWarper>();
    else if (!pp.algoComposition)
    {
        cout << "Projection inconnue '" << pp.surfaceCompo << "'\n";
        return 6;
    }

    pp.composition = pp.algoComposition->create(static_cast<float>(pp.focaleMoyenne));
    vector<Mat> mesmasques(nbImages);
    for (int i = 0; i < nbImages; ++i)
    {
        Mat_<float> K;
        pp.cameras[i].K().convertTo(K, CV_32F);

        pp.posCoin[i] = pp.composition->warp(matUSB[pp.indices[i]], K, pp.cameras[i].R, INTER_LINEAR, BORDER_REFLECT, imagesProjetees[i]);
        pp.tailleMasque[i] = imagesProjetees[i].size();
        pp.composition->warp(masques[i], K, pp.cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, pp.masqueCompo[i]);
        pp.masqueCompo[i].copyTo(mesmasques[i]);
    }
    pp.algoCorrectExpo = ExposureCompensator::createDefault(pp.correctionExposition);
    pp.algoCorrectExpo->feed(pp.posCoin, imagesProjetees, pp.masqueCompo);
    Ptr<SeamFinder> rechercheCouture;
    if (pp.typeCouture == "no")
        rechercheCouture = makePtr<detail::NoSeamFinder>();
    else if (pp.typeCouture == "voronoi")
        rechercheCouture = makePtr<detail::VoronoiSeamFinder>();
    else if (pp.typeCouture == "gc_color")
    {
        rechercheCouture = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
    }
    else if (pp.typeCouture == "gc_colorgrad")
    {
        rechercheCouture = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
    }
    else if (pp.typeCouture == "dp_color")
        rechercheCouture = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
    else if (pp.typeCouture == "dp_colorgrad")
        rechercheCouture = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
    if (!rechercheCouture)
    {
        cout << "type de couture inconnue :" << pp.typeCouture << "\n";
        return 7;
    }
    rechercheCouture->find(imagesProjetees, pp.posCoin, pp.masqueCompo);
    vector<Mat> mescoutures(nbImages);
    for (int i = 0; i < nbImages; ++i)
    {
        pp.masqueCompo[i].copyTo(mescoutures[i]);
    }
    pp.algoCorrectExpo->getMatGains(pp.gains);
    pp.algoCorrectExpo->setUpdateGain(true);

    SauverParamPano(pp);
    masques.clear();
    return 0;
}

Mat ComposerPanorama(vector<Mat> matUSB, ParamPano &pp)
{
    Ptr<Blender> melangeur;
    int nbImages = pp.indices.size();

        
    for (int idxImage = 0; idxImage < nbImages; ++idxImage)
    {
        int indexImage = pp.indices[idxImage];
        Mat K;
        pp.cameras[idxImage].K().convertTo(K, CV_32F);
        Mat imgProjetee;
        pp.composition->warp(matUSB[idxImage], K, pp.cameras[idxImage].R, INTER_LINEAR, BORDER_REFLECT, imgProjetee);
        pp.algoCorrectExpo->apply(idxImage, pp.posCoin[idxImage], imgProjetee, pp.masqueCompo[idxImage]);

        if (pp.masquePiece[idxImage].empty())
        {
            UMat dilated_mask;

            if (pp.coutureVisible)
                erode(pp.masqueCompo[idxImage], dilated_mask, UMat());
            else
                dilate(pp.masqueCompo[idxImage], dilated_mask, UMat());
            resize(dilated_mask, pp.masquePiece[idxImage], pp.masqueCompo[idxImage].size());
            bitwise_and(pp.masquePiece[idxImage], pp.masqueCompo[idxImage], dilated_mask);
            pp.masqueCompoPiece[idxImage] = dilated_mask;
        }
        if (!melangeur)
        {
            melangeur = Blender::createDefault(Blender::MULTI_BAND, false);
            Size rectDst = resultRoi(pp.posCoin, pp.tailleMasque).size();
            float blend_width = sqrt(static_cast<float>(rectDst.area())) * pp.forceMelange / 100.f;

            MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(melangeur.get());
            mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
            melangeur->prepare(pp.posCoin, pp.tailleMasque);
        }
        melangeur->feed(imgProjetee, pp.masqueCompoPiece[idxImage], pp.posCoin[idxImage]);
    }
    Mat result, result_mask;

    melangeur->blend(result, result_mask);
    if (result.type() == CV_16UC3 || result.type() == CV_16SC3)
        result.convertTo(result, CV_8U);
    return result;
}

void SauverParamPano(ParamPano pp)
{
    FileStorage fs("WebCamPanoramique.yml", FileStorage::WRITE);
    fs << "init" << (int)1;
    fs << "typeCouture" << pp.typeCouture;
    fs << "surfaceComposition" << pp.surfaceCompo;
    fs << "correctionExposition" << pp.correctionExposition;
    fs << "forceMelange" << pp.forceMelange;
    fs << "taille" << static_cast<int>(pp.posCoin.size());
    fs << "focaleMoyenne" << pp.focaleMoyenne;
    fs << "indices" << pp.indices;
    fs << "gainsize" << static_cast<int>(pp.gains.size());
    for (int i = 0; i < pp.gains.size(); i++)
    {
        fs << format("gain%d", i) << pp.gains[i];
    }
    for (int i = 0; i < pp.posCoin.size(); ++i)
    {
        fs << format("focal%d", i) << pp.focales[i];
        fs << format("tailleImage%d", i) << pp.tailleImages[i];
        fs << format("indice%d", i) << pp.indices[i];
        fs << format("coin%d", i) << pp.posCoin[i];
        fs << format("tailleMasque%d", i) << pp.tailleMasque[i];
        fs << format("masque%d", i) << pp.masqueCompo[i].getMat(ACCESS_READ);
        fs << format("cameraRot%d", i) << pp.cameras[i].R;
        fs << format("cameraFocal%d", i) << pp.cameras[i].focal;
        fs << format("cameraPPX%d", i) << pp.cameras[i].ppx;
        fs << format("cameraPPY%d", i) << pp.cameras[i].ppy;
    }
}

bool ChargerParamPano(ParamPano &pp)
{
    FileStorage fs("WebCamPanoramique.yml", FileStorage::READ);
    if (!fs.isOpened())
        return false;
    pp.init = (bool)((int)fs["init"]);
    pp.typeCouture = fs["typeCouture"];
    pp.surfaceCompo = fs["surfaceComposition"];
    pp.correctionExposition = fs["correctionExposition"];
    pp.forceMelange = fs["forceMelange"];
    int taille = fs["taille"];
    fs["focaleMoyenne"] >> pp.focaleMoyenne;
    fs["indices"] >> pp.indices;
    for (int i = 0; i < taille; ++i)
    {
        Size s;
        UMat ux;
        double xx;
        int j;
        fs[format("tailleImage%d", i)] >> s; pp.tailleImages.push_back(s);
        fs[format("focal%d", i)] >> xx; pp.focales.push_back(xx);
        fs[format("coin%d", i)] >> s; pp.posCoin.push_back(s);
        fs[format("tailleMasque%d", i)] >> s; pp.tailleMasque.push_back(s);
        fs[format("indice%d", i)] >> j;
        Mat x;
        fs[format("masque%d", i)] >> x;
        x.copyTo(ux);
        pp.masqueCompo.push_back(ux);
        pp.masquePiece.push_back(UMat());
        detail::CameraParams c;
        fs[format("cameraRot%d", i)] >> c.R;
        fs[format("cameraFocal%d", i)] >> c.focal;
        fs[format("cameraPPX%d", i)] >> c.ppx;
        fs[format("cameraPPY%d", i)] >> c.ppy;
        pp.cameras.push_back(c);
    }
    fs["gainsize"] >> taille;
    pp.gains.resize(taille);
    for (int i = 0; i < taille; i++)
    {
        fs[format("gain%d", i)] >> pp.gains[i];
    }
    pp.masqueCompoPiece.resize(pp.masquePiece.size());
    if (pp.surfaceCompo == "plane")
        pp.algoComposition = makePtr<cv::PlaneWarper>();
    else if (pp.surfaceCompo == "cylindrical")
        pp.algoComposition = makePtr<cv::CylindricalWarper>();
    else if (pp.surfaceCompo == "spherical")
        pp.algoComposition = makePtr<cv::SphericalWarper>();
    else if (pp.surfaceCompo == "fisheye")
        pp.algoComposition = makePtr<cv::FisheyeWarper>();
    else if (pp.surfaceCompo == "stereographic")
        pp.algoComposition = makePtr<cv::StereographicWarper>();
    pp.composition = pp.algoComposition->create(static_cast<float>(pp.focaleMoyenne));
    pp.algoCorrectExpo = ExposureCompensator::createDefault(pp.correctionExposition);
    pp.algoCorrectExpo->setMatGains(pp.gains);
    pp.algoCorrectExpo->setUpdateGain(false);
    return true;
}

vector<VideoCapture> RechercheCamera()
{
    vector<VideoCapture> v;

    for (int i =0; i<NBCAMERA; i++)
    {
        VideoCapture video;
        video.open(i + CAP_DSHOW);
        if (!video.isOpened())
        {
            cout << " cannot openned camera : " << i << endl;
        }
        else
        {
            video.set(CAP_PROP_FRAME_WIDTH, 640);
            video.set(CAP_PROP_FRAME_HEIGHT, 480);
            v.push_back(video);
            cout << "Camera : " << i << "-> " << video.get(CAP_PROP_FRAME_HEIGHT);
            cout << "  " << video.get(CAP_PROP_FRAME_WIDTH) << endl;
        }
    }
    return v;
}

vector<Mat> LireImagesSynchro(vector<ParamCamera> *pc)
{
    vector<Mat> x;
    int64 tps;
    int essai = 0;
    do
    {
        for (int i = 0; i<pc->size(); i++)
            mtxTimeStamp[i].lock();
        tps = static_cast<int64>(getTickCount() + getTickFrequency() * delai);
        for (int i = 0; i < pc->size(); i++)
        {
            (*pc)[i].captureImage = 1;
            (*pc)[i].tpsCapture = tps;
            mtxTimeStamp[i].unlock();
        }
        int64 sleepTime = static_cast<int64>(1.5*delai * 1000000);
        std::this_thread::sleep_for(std::chrono::microseconds(sleepTime));
        for (int i = 0; i < pc->size(); i++)
        {
            mtxTimeStamp[i].lock();
        }
        int nbCapture = 0;
        for (int i = 0; i < pc->size(); i++)
        {
            if ((*pc)[i].captureImage == 0)
            {
                nbCapture++;
            }
            else
                (*pc)[i].captureImage = 0;
        }
        for (int i = 0; i < pc->size(); i++)
        {
            if (nbCapture == pc->size())
            {
                x.push_back((*pc)[i].imAcq);
            }
            else
                (*pc)[i].captureImage = 0;
            mtxTimeStamp[i].unlock();
        }
        if (nbCapture<pc->size())
            delai += 0.03;
    } while (x.size() != pc->size() && essai <4);
    delai -= 0.015;
    return x;
}

vector<Mat> LireImages(vector<ParamCamera> *pc)
{
    vector<Mat> x(pc->size());
    for (int i = 0; i < pc->size(); i++)
    {
        mtxTimeStamp[i].lock();
        if (!(*pc)[i].derniereImage.empty())
            (*pc)[i].derniereImage.copyTo(x[i]);
        else
        {

            x.clear();
            mtxTimeStamp[i].unlock();
            return x;
        }
        mtxTimeStamp[i].unlock();
    }
    return x;
}

void GestionCmdCamera(ParamCamera *pc)
{
    double x = pc->index;
    switch (pc->cmd & 0xFF)
    {
    case 'g':
        x = pc->v->get(CAP_PROP_GAIN) - 1;
        pc->v->set(CAP_PROP_GAIN, x);
        break;
    case 'G':
        x = pc->v->get(CAP_PROP_GAIN) + 1;
        pc->v->set(CAP_PROP_GAIN, x);
        break;
    case 'b':
        x = pc->v->get(CAP_PROP_BRIGHTNESS) - 1;
        pc->v->set(CAP_PROP_BRIGHTNESS, x);
        break;
    case 'B':
        x = pc->v->get(CAP_PROP_BRIGHTNESS) + 1;
        pc->v->set(CAP_PROP_BRIGHTNESS, x);
        break;
    case 'E':
        x = pc->v->get(CAP_PROP_EXPOSURE) + 1;
        pc->v->set(CAP_PROP_EXPOSURE, x);
        break;
    case 'e':
        x = pc->v->get(CAP_PROP_EXPOSURE) - 1;
        pc->v->set(CAP_PROP_EXPOSURE, x);
        break;
    case 'c':
        x = pc->v->get(CAP_PROP_SATURATION) - 1;
        pc->v->set(CAP_PROP_SATURATION, x);
        break;
    case 'C':
        x = pc->v->get(CAP_PROP_SATURATION) + 1;
        pc->v->set(CAP_PROP_SATURATION, x);
        break;
    case 'w':
        pc->v->set(CAP_PROP_SETTINGS, x);
        break;
    }
}

void AcquisitionVideo(ParamCamera *pc)
{
    cout << "Running thread " << pc->index << endl;
    int64  tpsFrame = 0, tpsFramePre;
    Mat frame;
    *(pc->v) >> frame;
    if (frame.empty())
    {
        cout << "Image vide index ->" << pc->index << endl;
    }
    for (;;)
    {
        tpsFramePre = getTickCount();
        mtxTimeStamp[pc->index].lock();
        *(pc->v) >> frame;
        pc->derniereImage = frame;
        if (frame.empty())
            cout << "Image vide";
        tpsFrame = getTickCount();
        GestionCmdCamera(pc);
        if (stopThread)
            break;
        if (pc->captureImage)
        {
            if (!frame.empty() && tpsFrame >= pc->tpsCapture)
            {
                pc->captureImage = 0;
                frame.copyTo(pc->imAcq);
                pc->debAcq = tpsFramePre;
                pc->finAcq = tpsFrame;
            }
        }
        mtxTimeStamp[pc->index].unlock();
        tpsFrame = getTickCount();
        int64 sleepTime = static_cast<int64>((tpsFrame - tpsFramePre) / getTickFrequency() * 1000000);
        std::this_thread::sleep_for(std::chrono::microseconds(sleepTime));
    }
    mtxTimeStamp[pc->index].unlock();
}
