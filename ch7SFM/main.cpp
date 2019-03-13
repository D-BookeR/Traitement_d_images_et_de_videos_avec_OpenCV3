#include <opencv2/opencv.hpp> 
#include <opencv2/sfm.hpp> 
#include <opencv2/sfm/simple_pipeline.hpp> 
#ifdef HAVE_OPENCV_VIZ
#include <opencv2/viz.hpp>
#include <opencv2/viz/widget_accessor.hpp>
#endif
#include <thread>        
#include <mutex> 

using namespace cv;
using namespace std;
using namespace cv::sfm;

#define LARGEUR_ECRAN 1200
#define HAUTEUR_ECRAN 900
#define NBCAMERA 1
#define MODE_AFFICHAGE 0x100
#define MODE_REGLAGECAMERA 0x1000
#define OPTIM_INTRINSEQUE SFM_REFINE_FOCAL_LENGTH+ SFM_REFINE_PRINCIPAL_POINT
#define OPTIM_DISTORSION SFM_REFINE_RADIAL_DISTORTION_K1+ SFM_REFINE_RADIAL_DISTORTION_K2

struct ParamCalibration {
    int indexUSB=0;
    int index;
    Mat cameraMatrix, distCoeffs;
    Size tailleImage;
    Mat mapx;
    Mat mapy;
};

struct ParamCamera {
    VideoCapture *v;
    ParamCalibration *pc;
    int64 tpsGlobal;
    int tpsCapture;
    int captureImage;
    int cmd;
    double tempsDelai;
    vector<Mat> imAcq;
    vector<int64> debAcq, finAcq;
    Mat derniereImage;
};

struct ImageSFM {
    vector<String > nomImages;
    ParamCalibration pc;
    bool utilIntrinDist=false;
    bool imageCorrigee=true;
    int paramOptim= SFM_REFINE_FOCAL_LENGTH+ SFM_REFINE_PRINCIPAL_POINT+ SFM_REFINE_RADIAL_DISTORTION_K1+ SFM_REFINE_RADIAL_DISTORTION_K2;
    vector<Mat> ptCommun3d;
    Matx33d K;
    double  k1 = 0, k2 = 0, k3 = 0,p1=0,p2=0;
    vector<Mat> rotation, translation;
    vector<Mat> imageMouvement;
    int indImageSelect=-1;
};

vector<mutex> mtxTimeStamp(1);
int stopThread = 0;

static vector<Mat> LireImages(vector<ParamCamera> *, vector<VideoCapture> *, vector<Mat> &);
static void videoacquire(ParamCamera *);
static Mat zoom(Mat , float );
static Point TracerBouton(Mat , Rect , String );
static bool ChargerConfiguration(String , vector<ParamCalibration> &);
static void AjouterImageSfm(ImageSFM &, Mat );
static bool EffacerImageSFM(ImageSFM &);
static void CalculSFM(ImageSFM &);
static void VizSfm(ImageSFM &);
static void LireConfigSFM(String ,ImageSFM &);
static Mat AfficheListe(ImageSFM &);
static void SauverListe(ImageSFM &);
static void AfficherSelectionSFM(ImageSFM &);
static void SauverStructure(String, ImageSFM &);

int main(int argc, char **argv)
{
    cout << getBuildInformation() << "\n";
    vector<thread> th;
    vector<ParamCamera> pCamera(1);
    vector<ParamCalibration> pc(pCamera.size());
    vector<VideoCapture> v(pCamera.size());
    ChargerConfiguration("config.yml", pc);
    Size tailleGlobale(0, 0);
    vector<Size> tailleWebcam(1);
    float zoomAffichage = 1;
    ImageSFM listeSFM;
    vector<Mat> x;
    int modeAffichage = 0;
    int code = 0;
    int indImage = 0, cameraSelect = 0;

    listeSFM.pc = pc[0];
    int64 tpsGlobal = getTickCount();
    for (int i = 0; i<pc.size(); i++)
    {
        VideoCapture vid(pc[i].indexUSB);
        if (vid.isOpened())
        {
            v[i] = vid;
            vid.set(CAP_PROP_FRAME_WIDTH, pc[i].tailleImage.width);
            vid.set(CAP_PROP_FRAME_HEIGHT, pc[i].tailleImage.height);
            tailleWebcam[i] = Size(v[i].get(CAP_PROP_FRAME_WIDTH), v[i].get(CAP_PROP_FRAME_HEIGHT));
            pc[i].tailleImage = tailleWebcam[i];
            if (i == 0)
                tailleGlobale = tailleWebcam[i];
            else if (i % 2 == 1)
                tailleGlobale.width += tailleWebcam[i].width;
            else
                tailleGlobale.height += tailleWebcam[i].height;
            pCamera[i].v = &v[i];
            pCamera[i].pc = &pc[i];
            pCamera[i].cmd = MODE_AFFICHAGE;
            pCamera[i].tpsGlobal = tpsGlobal;
            pCamera[i].tempsDelai = 0.1;

            thread t(videoacquire, &pCamera[i]);
            t.detach();
        }
        else
            th.push_back(thread());
    }
    if (tailleGlobale.width>LARGEUR_ECRAN || tailleGlobale.height>HAUTEUR_ECRAN)
        zoomAffichage = min(LARGEUR_ECRAN / float(tailleGlobale.width), HAUTEUR_ECRAN / float(tailleGlobale.height));

    imshow("SFM liste", AfficheListe(listeSFM));

    do
    {
        code = waitKey(1);
        if (modeAffichage&MODE_REGLAGECAMERA)
        {
            switch (code) {
            case '0':
                mtxTimeStamp[0].lock();
                pCamera[0].cmd = pCamera[0].cmd | (MODE_REGLAGECAMERA);
                mtxTimeStamp[0].unlock();
                cameraSelect = 0;
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
            case 'R':
                modeAffichage = modeAffichage& (~MODE_REGLAGECAMERA);
                mtxTimeStamp[cameraSelect].lock();
                pCamera[cameraSelect].cmd = pCamera[cameraSelect].cmd & (~MODE_REGLAGECAMERA);
                mtxTimeStamp[cameraSelect].unlock();
                break;
            }
        }
        else
        {
            Mat dst;
            switch (code) {
            case 'a':
            case 'A':
                undistort(x[0], dst, pc[0].cameraMatrix, pc[0].distCoeffs);
                listeSFM.nomImages.push_back(format("img%d.png", indImage++));
                imwrite(listeSFM.nomImages[listeSFM.nomImages.size() - 1], dst);
                imshow("Image SFM", dst);
                waitKey(30);
                AjouterImageSfm(listeSFM, dst);
                listeSFM.indImageSelect = static_cast<int>(listeSFM.nomImages.size()) - 1;
                AfficherSelectionSFM(listeSFM);
                if (code == 'A')
                {
                    CalculSFM(listeSFM);
                    if (listeSFM.ptCommun3d.size()>0)
                        VizSfm(listeSFM);
                }

                imshow("SFM liste", AfficheListe(listeSFM));
                break;
            case 'c':
                CalculSFM(listeSFM);
                if (listeSFM.ptCommun3d.size()>0)
                    VizSfm(listeSFM);
                SauverListe(listeSFM);
                break;
            case 'd':
                listeSFM.imageCorrigee = !listeSFM.imageCorrigee;
                if (listeSFM.imageCorrigee)
                    cout << "Image corrigée\n";
                else
                    cout << "Image non corrigée\n";
                break;
            case 'e':
                if (EffacerImageSFM(listeSFM) && listeSFM.indImageSelect >= 0)
                    AfficherSelectionSFM(listeSFM);
                else
                    destroyWindow("Image SFM");
                imshow("SFM liste", AfficheListe(listeSFM));
                break;
            case 'i':
                listeSFM.utilIntrinDist = !listeSFM.utilIntrinDist;
                if (listeSFM.utilIntrinDist )
                    cout << "Valeurs initiales à partir de la caméra\n";
                else
                    cout << "Valeurs initiales à partir du résultat \n";
                imshow("SFM liste", AfficheListe(listeSFM));
                break;
            case 'l':
                LireConfigSFM("ImagesSFM1.yml",listeSFM);
                imshow("SFM liste", AfficheListe(listeSFM));
                indImage= static_cast<int>(listeSFM.nomImages.size());
                AfficherSelectionSFM(listeSFM);
                break;
            case 'o':
                if (listeSFM.paramOptim & OPTIM_INTRINSEQUE)
                    listeSFM.paramOptim = (~OPTIM_INTRINSEQUE) & listeSFM.paramOptim;
                else
                    listeSFM.paramOptim = (OPTIM_INTRINSEQUE) | listeSFM.paramOptim;

                if (listeSFM.paramOptim & OPTIM_INTRINSEQUE)
                    cout << "Optimisation parametres intrinsèques\n";
                else
                    cout << "Parametres intrinsèques fixés\n";
                break;
            case 'p':
                if (listeSFM.indImageSelect >= 0)
                {
                    if (listeSFM.indImageSelect>0)
                        listeSFM.indImageSelect--;
                    else
                        listeSFM.indImageSelect = static_cast<int>(listeSFM.nomImages.size()) - 1;
                    AfficherSelectionSFM(listeSFM);
                }
                break;
            case 'R':
                modeAffichage = modeAffichage | (MODE_REGLAGECAMERA);
                mtxTimeStamp[cameraSelect].lock();
                pCamera[cameraSelect].cmd = pCamera[cameraSelect].cmd | MODE_REGLAGECAMERA;
                mtxTimeStamp[cameraSelect].unlock();
                break;
            case 's':
                if (listeSFM.indImageSelect>=0)
                {
                    if (listeSFM.indImageSelect<listeSFM.nomImages.size() - 1)
                        listeSFM.indImageSelect++;
                    else
                        listeSFM.indImageSelect=0;
                    AfficherSelectionSFM(listeSFM);
                }
                break;
            case 'u':
                if (listeSFM.paramOptim & OPTIM_DISTORSION)
                    listeSFM.paramOptim = listeSFM.paramOptim & (~OPTIM_DISTORSION);
                else
                    listeSFM.paramOptim = listeSFM.paramOptim | OPTIM_DISTORSION;
                if (listeSFM.paramOptim & OPTIM_DISTORSION)
                    cout << "Optimisation des coefficients de distorsion\n";
                else
                    cout << "Coefficients de distorsion fixés\n";
                break;
            case 'v':
                VizSfm(listeSFM);
                break;
            case '3':
                SauverStructure("pt3d.yml", listeSFM);
                break;
            }
        }
        x = LireImages(&pCamera, &v, x);
        if (!x[0].empty())
            imshow(format("Webcam"), zoom(x[0], zoomAffichage));
    } 
    while (code != 27);
}

void GestionCmdCamera(ParamCamera *pc)
{
    double x = pc->pc->index;
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
    pc->cmd = pc->cmd & 0xFFFFFF00;
}


bool ChargerConfiguration(String nomFichierConfiguration, vector<ParamCalibration> &pc)
{
    FileStorage fs(nomFichierConfiguration, FileStorage::READ);
    pc[0].index = 0;
    pc[0].indexUSB = 0;
    if (fs.isOpened())
    {
        if (!fs["Cam0index"].empty())
            fs["Cam0index"] >> pc[0].indexUSB;
        if (!fs["Cam0Size"].empty())
            fs["Cam0Size"] >> pc[0].tailleImage;
        if (!fs["cameraMatrice0"].empty())
            fs["cameraMatrice0"] >> pc[0].cameraMatrix;
        if (!fs["cameraDistorsion0"].empty())
            fs["cameraDistorsion0"] >> pc[0].distCoeffs;
        return true;
    }
    else
    {
        FileStorage fs(nomFichierConfiguration, FileStorage::WRITE);
        pc[0].index = 0;
        pc[0].indexUSB = 0;
        if (fs.isOpened())
        {
            fs<<"Cam0index"<< pc[0].indexUSB;
            fs<<"Cam0Size"<<Size(640,480);
            fs<<"cameraMatrice0"<<(Mat_<double>(3,3)<<1,0,320,0,1,240,0,0,1);
            fs<<"cameraDistorsion0"<< (Mat_<double>(1, 5) << 0,0,0,0,0);;
            return true;
        }
    }
    return false;
}


void SauverListe(ImageSFM &lSFM)
{
    FileStorage fs("ImagesSFM.yml", FileStorage::WRITE);
    fs << "images" << "[";
    for (int i = 0; i < lSFM.nomImages.size(); i++) 
    {
        fs << lSFM.nomImages[i];
    }
    fs << "]";
    fs << "cameraMatrice"<< Mat(lSFM.K);
    fs << "cameraDistorsion_k1" << lSFM.k1;
    fs << "cameraDistorsion_k2" << lSFM.k2;
    fs << "cameraDistorsion_k3" << lSFM.k3;
    fs << "cameraDistorsion_p1" << lSFM.p1;
    fs << "cameraDistorsion_p2" << lSFM.p2;
    for (int i = 0; i < lSFM.rotation.size(); i++) 
    {
        fs << format("R%d", i) << lSFM.rotation[i];
        fs << format("T%d", i) << lSFM.translation[i];
    }
}

vector<Mat> LireImages(vector<ParamCamera> *pc, vector<VideoCapture> *v, vector<Mat> &x)
{
    vector<Mat> xx;
    if (x.size() != v->size())
        xx.resize(v->size());
    else
        xx = x;
    for (int i = 0; i < v->size(); i++)
    {
        mtxTimeStamp[i].lock();
        (*pc)[i].derniereImage.copyTo(xx[i]);
        mtxTimeStamp[i].unlock();
    }
    return xx;
}

Mat zoom(Mat x, float w)
{
    if (w != 1)
    {
        Mat y;
        resize(x, y, Size(), w, w);
        return y;
    }
    return x;
}


void ClavierViz(const viz::KeyboardEvent   	&k,void *c)
{
    int *i=(int*)c;
    if (k.action== cv::viz::KeyboardEvent::Action::KEY_UP)
    {
        if (k.code=='1')
            *i=*i-1;
        else if (k.code == '2')
            *i = *i + 1;
    }
}


void VizSfm(ImageSFM &lImg)
{
    if (lImg.ptCommun3d.size()==0)
    {
        cout<< "Pas de points 3D\n";
        return;
    }
    else
        cout<< lImg.ptCommun3d.size()<<" points 3D\n";
#ifdef HAVE_OPENCV_VIZ
    vector<Vec3f> point3d;
    vector<Vec3b> couleur;
    viz::Viz3d fen3D("Monde");
    fen3D.setBackgroundColor(viz::Color::black());
    vector<Affine3d> path;
    vector<Mat> poseCamera(lImg.rotation.size());
    for (size_t i = 0; i < lImg.rotation.size(); ++i)
    {
        poseCamera[i]=Mat::eye(4,4,CV_64F);
        lImg.rotation[i].copyTo(poseCamera[i](Range(0, 3), Range(0, 3)));
        lImg.translation[i].copyTo(poseCamera[i](Range(0, 3), Range(3, 4)));
        path.push_back(Affine3d(lImg.rotation[i], lImg.translation[i]).inv());
    }
    for (int i = 0; i < lImg.ptCommun3d.size(); ++i)
    {
        point3d.push_back(Vec3f(lImg.ptCommun3d[i]));
        Mat p1, p2,p3;
        convertPointsToHomogeneous(lImg.ptCommun3d[i].t(), p1);
        p1 = p1.reshape(1).t();
        int nbPixel=0;
        Vec3f couleurPixel(0,0,0);
        for (int j = 0; j < poseCamera.size(); j++)
        {
            p2 = poseCamera[j] * p1;
            p2.convertTo(p3, CV_64F);
            p3 = p3.rowRange(Range(0, 3));
            p2 = Mat(lImg.K) * p3;
            convertPointsFromHomogeneous(p2.t(), p3);
            Point pt(p3.at<double>(0, 0), p3.at<double>(0, 1));
            if (pt.x >= 0 && pt.x<lImg.imageMouvement[0].cols &&
                pt.y >= 0 && pt.y<lImg.imageMouvement[0].rows)
            {
                couleurPixel += lImg.imageMouvement[0].at<Vec3b>(pt);
                nbPixel++;
            }

        }
        if (nbPixel==0)
            couleur.push_back(viz::Color(viz::Color::red()));
        else
            couleur.push_back(couleurPixel/nbPixel);
    }
    int indCam=0,indRef=-1;
    viz::WCameraPosition orientationCamera(0.01); 
    viz::WCloud nuage(point3d, couleur);
    
    fen3D.registerKeyboardCallback(ClavierViz, &indCam);
    do 
    {
        if (indRef!=indCam)
        {
            if (indCam <0)
                indCam = 0;
            else if (indCam>= path.size())
                indCam= path.size() - 1;
            indRef=indCam; lImg.indImageSelect= indRef;
            AfficherSelectionSFM(lImg);
            fen3D.removeAllWidgets();
            viz::WCameraPosition planCamera(lImg.K, lImg.imageMouvement[indCam], 0.03, viz::Color::bluberry()); // Camera frustum
            fen3D.setViewerPose(path[indCam]);
            fen3D.showWidget("Nuage", nuage);
            fen3D.showWidget("OrientationCamera", orientationCamera, path[indCam]);
            fen3D.showWidget("PlanCamera", planCamera, path[indCam]);
            fen3D.showWidget("Orientation", viz::WTrajectory(path, viz::WTrajectory::PATH, 0.1, viz::Color(viz::Color::yellow())));
        }
        fen3D.spinOnce(1, true);

    } while (!fen3D.wasStopped());
#else
    cout<< "VTK ou viz sont absents\n";
#endif
}
void SauverStructure(String nomFichier,ImageSFM &lImg)
{
    vector<Point3f> ptReconstruct;
    for (int i = 0; i < lImg.ptCommun3d.size(); ++i)
    {
        ptReconstruct.push_back(Vec3f(lImg.ptCommun3d[i]));
    }
    FileStorage f(nomFichier, FileStorage::WRITE);
    f << "pt3d" << ptReconstruct;
    f << "pt3dOrig" << lImg.ptCommun3d;
    ofstream fs("ptCommun.txt", ios_base::app);
    if (fs.is_open())
    {
        for (int i = 0; i < lImg.ptCommun3d.size(); ++i)
        {
            fs<<ptReconstruct[i].x<<"\t"<< ptReconstruct[i].y << "\t"<<ptReconstruct[i].z << "\n";
        }
    }

}

void videoacquire(ParamCamera *pc)
{
    int64 tpsInit = getTickCount();
    int64 tckPerSec = static_cast<int64> (getTickFrequency());
    int modeAffichage = 0;
    int64  tpsFrame = 0, tpsFramePre;
    int64 tpsFrameAsk;//, periodeTick = static_cast<int64> (getTickFrequency() / pc->fps);
    Mat frame, cFrame;
    Mat     frame1;
    Mat     frame2;

    *(pc->v) >> frame;
    frame.copyTo(frame1);
    frame.copyTo(frame2);
    tpsFrame = getTickCount();
    int64 offsetTime = pc->tpsGlobal + 2 * getTickFrequency();
    do
    {
        tpsFrame = getTickCount();
    } while (tpsFrame<offsetTime);
    tpsFrameAsk = offsetTime;
    int i = 0, index = pc->pc->index;
    for (int nbAcq = 0;;)
    {
        tpsFramePre = getTickCount();
        *(pc->v) >> frame;

        tpsFrame = getTickCount();
        nbAcq++;
        int64 dt = tpsFrame - tpsFrameAsk;
        mtxTimeStamp[pc->pc->index].lock();
        if (modeAffichage&MODE_REGLAGECAMERA)
            GestionCmdCamera(pc);
        if (stopThread)
            break;
        if (pc->captureImage)
        {
            if (tpsFrame >= pc->tpsCapture)
            {
                pc->captureImage = 0;
                pc->imAcq.push_back(frame.clone());
                pc->debAcq.push_back(tpsFramePre);
                pc->finAcq.push_back(tpsFrame);
            }
        }
        if (pc->cmd & MODE_AFFICHAGE)
            modeAffichage = modeAffichage | MODE_AFFICHAGE;
        else
            modeAffichage = modeAffichage & ~MODE_AFFICHAGE;
        if (pc->cmd & MODE_REGLAGECAMERA)
            modeAffichage = modeAffichage | MODE_REGLAGECAMERA;
        else
            modeAffichage = modeAffichage & ~MODE_REGLAGECAMERA;
        mtxTimeStamp[pc->pc->index].unlock();

        if (modeAffichage & MODE_AFFICHAGE)
        {
            mtxTimeStamp[pc->pc->index].lock();
            frame.copyTo(pc->derniereImage);
            mtxTimeStamp[pc->pc->index].unlock();
        }

        tpsFrame = getTickCount();
    }
    mtxTimeStamp[pc->pc->index].unlock();
}

void CalculSFM(ImageSFM &lSFM)
{
#ifdef HAVE_OPENCV_SFM
    if (lSFM.utilIntrinDist)
    {
        lSFM.K = lSFM.pc.cameraMatrix.clone();
        if (!lSFM.imageCorrigee)
        {
            lSFM.k1 = lSFM.pc.distCoeffs.at<double>(0, 0);
            lSFM.k2 = lSFM.pc.distCoeffs.at<double>(0, 1);
            lSFM.k3 = lSFM.pc.distCoeffs.at<double>(0, 4);
            lSFM.p1 = lSFM.pc.distCoeffs.at<double>(0, 2);
            lSFM.p2 = lSFM.pc.distCoeffs.at<double>(0, 3);
        }
    }
    else
    {
        if (!lSFM.imageCorrigee)
        {
            lSFM.k1 = 0;
            lSFM.k2 = 0;
            lSFM.k3 = 0;
            lSFM.p1 = 0;
            lSFM.p2 = 0;
        }
    }
    

    libmv_ReconstructionOptions optionsReconstruction(0, 1, lSFM.paramOptim, 0, -1);
    libmv_CameraIntrinsicsOptions optionsModeleCamera =
        libmv_CameraIntrinsicsOptions(SFM_DISTORTION_MODEL_POLYNOMIAL,
            lSFM.K(0, 0), lSFM.K(0, 2), lSFM.K(1, 2),
            lSFM.k1, lSFM.k2, lSFM.k3, lSFM.p1, lSFM.p2);
    Ptr<BaseSFM> reconstruction =SFMLibmvEuclideanReconstruction::create(optionsModeleCamera, optionsReconstruction);
    if (lSFM.nomImages.size() > 1 )
    {
/*        try
        {
            vector<string> s;
            for (int i = 0; i<lSFM.nomImages.size(); i++)
                s.push_back(lSFM.nomImages[i]);

            reconstruction->run(s, lSFM.K, lSFM.rotation, lSFM.translation, lSFM.ptCommun3d);
        }
        catch (cv::Exception e)
        {*/
            reconstruction->run(lSFM.nomImages, lSFM.K, lSFM.rotation, lSFM.translation, lSFM.ptCommun3d);
 //       }
        lSFM.k1 = optionsModeleCamera.polynomial_k1;
        lSFM.k2 = optionsModeleCamera.polynomial_k2;
        lSFM.k3 = optionsModeleCamera.polynomial_k3;
        lSFM.p1 = optionsModeleCamera.polynomial_p1;
        lSFM.p2 = optionsModeleCamera.polynomial_p2;
    }
#else
    cout << "Le module SFM n'est pas dsponible dans votre configuration\n";
#endif
}

void AjouterImageSfm(ImageSFM &lSFM, Mat dst)
{
    if (lSFM.imageMouvement.size() == 0)
        lSFM.K  << 1, 0, dst.cols / 2, 0, 1, dst.rows / 2, 0, 0, 1;
    lSFM.imageMouvement.push_back(dst);
    SauverListe(lSFM);
}


void AfficherSelectionSFM(ImageSFM &lSFM)
{
    if (lSFM.nomImages.size() == 0)
    {
        cout << "Attention liste image vide_n";
        return;
    }
    Mat dst = imread(lSFM.nomImages[lSFM.indImageSelect], IMREAD_COLOR);
    TracerBouton(dst, Rect(Point(10, 50), Size(300, 60)), lSFM.nomImages[lSFM.indImageSelect]);
    imshow("Image SFM", dst);
    waitKey(10);
}


Point TracerBouton(Mat img, Rect r, String s)
{
    int nomPolice = FONT_HERSHEY_SCRIPT_SIMPLEX;
    double echellePolice = 1;
    int epaisseurTrait = 1;
    int ligneBase = 0;
    Size ts = getTextSize(s, nomPolice, echellePolice, epaisseurTrait, &ligneBase);
    ts = r.size() - ts;
    Point p(r.tl().x, r.br().y);
    if (ts.height>0)
        p.y += -ts.height / 2 - 6;
    if (ts.width>0)
        p.x += ts.width / 2;
    putText(img, s, p, nomPolice, echellePolice, Scalar(255, 255, 255), epaisseurTrait);
    rectangle(img, r, Scalar(128, 128, 128), 2);
    return p;
}

bool EffacerImageSFM(ImageSFM &lSFM)
{
    int indice = lSFM.indImageSelect;
    if (indice<0 || indice >= lSFM.imageMouvement.size())
        return false;
    lSFM.nomImages.erase(lSFM.nomImages.begin() + indice);
    lSFM.imageMouvement.erase(lSFM.imageMouvement.begin() + indice);
    if (lSFM.indImageSelect >= lSFM.nomImages.size())
        lSFM.indImageSelect = static_cast<int>(lSFM.nomImages.size()) - 1;
    return true;
}

void LireConfigSFM(String s,ImageSFM &lSFM)
{

    FileStorage fs(s, FileStorage::READ);
    if (fs.isOpened())
    {
        lSFM.nomImages.clear();
        lSFM.imageMouvement.clear();
        if (!fs["images"].empty())
            fs["images"] >> lSFM.nomImages;
        if (lSFM.nomImages.empty())
            lSFM.indImageSelect = -1;
        else
            lSFM.indImageSelect = 0;
        for (int i = 0; i < lSFM.nomImages.size(); i++)
        {
            Mat x = imread(lSFM.nomImages[i], IMREAD_COLOR);
            lSFM.imageMouvement.push_back(x);
        }
        for (int i = 0; i < lSFM.nomImages.size(); i++)
        {
            if (lSFM.imageMouvement[i].empty())
            {
                lSFM.indImageSelect = i;
                EffacerImageSFM(lSFM);
            }
        }
        if (!fs["cameraMatrice"].empty())
        {
            fs["cameraMatrice"] >> lSFM.pc.cameraMatrix;
            fs["cameraDistorsion"] >> lSFM.pc.distCoeffs;
            lSFM.K = lSFM.pc.cameraMatrix;
            lSFM.k1 = lSFM.pc.distCoeffs.at<double>(0, 0);
            lSFM.k2 = lSFM.pc.distCoeffs.at<double>(0, 1);
            lSFM.k3 = lSFM.pc.distCoeffs.at<double>(0, 4);
            lSFM.p1 = lSFM.pc.distCoeffs.at<double>(0, 2);
            lSFM.p2 = lSFM.pc.distCoeffs.at<double>(0, 3);
        }
    }
}

Mat AfficheListe(ImageSFM &lSFM)
{
    int nomPolice = FONT_HERSHEY_SCRIPT_SIMPLEX;
    double echellePolice = 1;
    int epaisseurTrait = 1;
    int ligneBase = 5;
    Size ts = getTextSize("Liste videgp", nomPolice, echellePolice, epaisseurTrait, &ligneBase);
    ts = Size(ts.width * 2, ts.height * 2);
    if (lSFM.nomImages.size() == 0)
    {
        Mat dst(ts.height * 2, ts.width + 100, CV_8UC3, Scalar(0));
        putText(dst, "Liste vide", Point(20, ts.height), nomPolice, echellePolice, Scalar(255, 255, 255), epaisseurTrait);
        return dst;
    }
    //    Size ts=getTextSize(listeSFM.nomImages[0], nomPolice, echellePolice, epaisseurTrait, &ligneBase);
    Mat dst(11 * (ts.height + 10), 5 * ts.width, CV_8UC3, Scalar(0));
    for (int i = 0; i < min(20, static_cast<int>(lSFM.nomImages.size())); i++)
    {
        Point p = TracerBouton(dst, Rect(Point(20, (i + 1)*(ts.height + 10)), ts), lSFM.nomImages[i]);
    }
    return dst;
}

