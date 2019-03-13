#include <opencv2/opencv.hpp> 
#include <opencv2/stereo.hpp> 
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>

#define LARGEUR_ECRAN 1200
#define HAUTEUR_ECRAN 900

#ifdef USE_VTK
#include <opencv2/viz.hpp>
#include <opencv2/viz/widget_accessor.hpp>
#endif

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iterator>
#include <thread>        
#include <mutex> 
#include <inttypes.h>

using namespace cv;
using namespace std;

#define NBCAMERA 2
struct ParamAlgoStereo{
    Ptr<StereoSGBM> sgbm;
    Ptr<StereoBM> bm;
    
    int typeAlgoSGBM;
    int typePreFiltre= StereoBM::PREFILTER_XSOBEL;
    int tailleBloc=3;
    int nbDisparite=8;
    int etenduSpeckle=1;
    int tailleSpeckle=10;
    int unicite=3;
    int preFilterCap=31;
};

struct ParamMire {
    int nbL = 9, nbC = 6;
    int nbLAruco = 5, nbCAruco = 8;
    float dimCarre = 0.0275;
    float dimAruco = 0.034625;
    float sepAruco = 0.02164;
    int dict = cv::aruco::DICT_7X7_250;
    Ptr<aruco::Dictionary> dictionary;
    Ptr<aruco::CharucoBoard> gridboard;
    Ptr<aruco::Board> board;
    Ptr<aruco::DetectorParameters> detectorParams;
};
struct ParamCalibration3D {
    vector<int> typeCalib3D = { CALIB_FIX_INTRINSIC + CALIB_ZERO_DISPARITY,CALIB_ZERO_DISPARITY };
    Mat R, T, E, F;
    Mat R1,R2,P1,P2;
    Mat Q;
    vector<Mat> m;
    vector<Mat> d;
    Rect valid1,valid2;
    double rms;
    String ficDonnees;
    vector<vector<Point2f>>  pointsCameraGauche;
    vector<vector<Point2f>>  pointsCameraDroite;
    vector<vector<Point3f> > pointsObjets;
};

struct SuiviDistance{
    ParamCalibration3D *pStereo;
    Mat disparite;
    Mat m;
    vector<Point3d> p;
    float zoomAffichage;
};

struct ParamCalibration {
    vector<int> typeCalib2D = { 0,CALIB_FIX_K4 + CALIB_FIX_K5 + CALIB_FIX_K6 + CALIB_ZERO_TANGENT_DIST,CALIB_FIX_K4 + CALIB_FIX_K5 + CALIB_FIX_K6+ CALIB_ZERO_TANGENT_DIST };
    int indexUSB;
    int index;
    vector<Mat> rvecs, tvecs;
    Mat cameraMatrix, distCoeffs;
    Mat mapx;
    Mat mapy;
    int nbImageGrille=0;
    vector<Point3f>  pointsGrille;
    vector<vector<Point2f>>  pointsCamera;
    vector<vector<Point3f> > pointsObjets;
    Size tailleImage;
    double rms;
    String ficDonnees;
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
    vector<int64> debAcq,finAcq;
    Mat derniereImage;
};


vector<mutex> mtxTimeStamp(NBCAMERA);
vector<vector<int64>> tps(mtxTimeStamp.size());

int stopThread=0;

static void videoacquire(ParamCamera *pc);
static vector<Mat> LireImages(vector<ParamCamera> *pc, vector<VideoCapture> *v,vector<Mat> &x);
static vector<Mat> LireImagesSynchro(vector<ParamCamera> *pc, vector<VideoCapture> *v);
static vector<VideoCapture> RechercheCamera();
static void AjouteGlissiere(String nomGlissiere, String nomFenetre, int minGlissiere, int maxGlissiere, int valeurDefaut, int *valGlissiere, void(*f)(int, void *), void *r = NULL);
static void MAJParamStereo(int x, void *r);
static void GestionCmdCamera(ParamCamera *pc);
static double ErreurDroiteEpipolaire(ParamCalibration3D sys3d);
static Mat zoom(Mat , float,SuiviDistance * = NULL);
static bool ChargerConfiguration(String nomFichierConfiguration, ParamMire &mire, vector<ParamCalibration> &pc, ParamCalibration3D &sys3d, ParamAlgoStereo &);
static void SauverConfiguration(String nomFichierConfiguration, ParamMire mire, vector<ParamCalibration> pc, ParamCalibration3D &sys3d,ParamAlgoStereo &);
static void SauverDonneesCalibration(ParamCalibration *pc1, ParamCalibration *pc2, ParamCalibration3D *sys3d);
static void ChargerDonneesCalibration(String nomFichier, ParamCalibration *pc, ParamCalibration3D *sys3d);
static bool AnalyseCharuco(Mat x,  ParamCalibration &pc, Mat frame, ParamMire &mire);
static bool AnalyseGrille(Mat x,  ParamCalibration *pc, ParamCalibration3D *sys3d, int index, Mat frame, ParamMire &mire);
#ifdef USE_VTK
static void VizMonde(Mat img,  Mat xyz);
static void VizDisparite(Mat img, Mat disp);
#endif
static void MesureDistance(int event, int x, int y, int flags, void *userdata);


#define MODE_AFFICHAGE 0x100
#define MODE_CALIBRE 0x200
#define MODE_MAPSTEREO 0x400
#define MODE_EPIPOLAIRE 0x800
#define MODE_REGLAGECAMERA 0x1000


int main (int argc,char **argv)
{

    ParamMire mire;
    int typeDistorsion =0;
    mire.dictionary =aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(mire.dict));
    mire.gridboard = aruco::CharucoBoard::create(mire.nbCAruco, mire.nbLAruco, mire.dimAruco, mire.sepAruco, mire.dictionary);
    mire.board = mire.gridboard.staticCast<aruco::Board>();
    mire.detectorParams = aruco::DetectorParameters::create();
    mire.detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
    mire.detectorParams->cornerRefinementMinAccuracy = 0.01;
    float zoomAffichage=1;

    vector<thread> th;
    vector<double> tpsMoyen;
    ParamCalibration3D sys3d;
    SuiviDistance sDistance;
    ParamAlgoStereo pStereo;
    vector<ParamCamera> pCamera(2);
    vector<ParamCalibration> pc(pCamera.size());
    vector<VideoCapture> v(pCamera.size());

    Size tailleGlobale(0,0);
    vector<Size> tailleWebcam(pCamera.size());
    bool configActive = ChargerConfiguration("config.yml", mire, pc, sys3d,pStereo);
    int64 tpsGlobal = getTickCount();
    int nbCamera=0;

    for (int i = 0; i<pc.size();i++)
    {
        VideoCapture vid(pc[i].indexUSB+CAP_DSHOW);
        if (vid.isOpened())
        {
            nbCamera++;
            v[i]=vid;
            vid.set(CAP_PROP_FRAME_WIDTH, pc[i].tailleImage.width);
            vid.set(CAP_PROP_FRAME_HEIGHT, pc[i].tailleImage.height);
            tailleWebcam[i]=Size(v[i].get(CAP_PROP_FRAME_WIDTH), v[i].get(CAP_PROP_FRAME_HEIGHT));
            pc[i].tailleImage = tailleWebcam[i];
            if (i==0)
                tailleGlobale= tailleWebcam[i];
            else if (i%2==1)
                tailleGlobale.width += tailleWebcam[i].width;
            else
                tailleGlobale.height += tailleWebcam[i].height;
            pCamera[i].v = &v[i];
            pCamera[i].pc = &pc[i];
            pCamera[i].cmd = 0;
            pCamera[i].tpsGlobal= tpsGlobal;
            pCamera[i].tempsDelai = 0.1;
            
            thread t(videoacquire, &pCamera[i]);
            t.detach();
        }
        else
            th.push_back(thread() );
    }
    if (nbCamera != 2)
    {
        cout<< nbCamera << " trouvée(s). ";
        cout<<"Il faut uniquement 2 caméras!\n";
        return 0;
    }
    if (!configActive)
        SauverConfiguration("config.yml", mire, pc, sys3d,pStereo);
    if (tailleGlobale.width>LARGEUR_ECRAN || tailleGlobale.height>HAUTEUR_ECRAN)
        zoomAffichage = min(LARGEUR_ECRAN/float(tailleGlobale.width) ,  HAUTEUR_ECRAN/float(tailleGlobale.height) );

    Mat frame(tailleGlobale,CV_8UC3,Scalar(0,0,0));
    Point centre(20,20);
    vector<int64> tCapture;
    vector<Mat> x;
    int modeAffichage=0;
    int algoStereo=0;

    Mat map11, map12, map21, map22;

    if (!sys3d.R1.empty())
        initUndistortRectifyMap(sys3d.m[0], sys3d.d[0], sys3d.R1, sys3d.P1, tailleWebcam[0], CV_16SC2, map11, map12);
    if (!sys3d.R2.empty())
        initUndistortRectifyMap(sys3d.m[1], sys3d.d[1], sys3d.R2, sys3d.P2, tailleWebcam[1], CV_16SC2, map21, map22);
    sDistance.pStereo =&sys3d;
    sDistance.zoomAffichage = zoomAffichage;
    imshow("Cameras",zoom(frame, zoomAffichage));
    namedWindow("Control",WINDOW_NORMAL);
    if (!map11.empty() && !map21.empty())
    {
        pStereo.bm = StereoBM::create(16*pStereo.nbDisparite, 2 * pStereo.tailleBloc + 1);
        pStereo.sgbm = StereoSGBM::create(0, 16*pStereo.nbDisparite, 2 * pStereo.tailleBloc + 1);
        pStereo.bm->setPreFilterType(pStereo.typePreFiltre);
        pStereo.bm->setUniquenessRatio(pStereo.unicite);
        pStereo.sgbm->setUniquenessRatio(pStereo.unicite);
        pStereo.bm->setSpeckleWindowSize(pStereo.tailleSpeckle);
        pStereo.sgbm->setSpeckleWindowSize(pStereo.tailleSpeckle);
        pStereo.bm->setSpeckleRange(pStereo.etenduSpeckle);
        pStereo.sgbm->setSpeckleRange(pStereo.etenduSpeckle);
        AjouteGlissiere("Bloc", "Control", 2, 100, pStereo.tailleBloc, &pStereo.tailleBloc, MAJParamStereo, &pStereo);
        AjouteGlissiere("nbDisparite", "Control", 1, 100, pStereo.nbDisparite, &pStereo.nbDisparite, MAJParamStereo, &pStereo);
        AjouteGlissiere("Unicite", "Control", 3, 100, pStereo.unicite, &pStereo.unicite, MAJParamStereo, &pStereo);
        AjouteGlissiere("EtenduSpeckle", "Control", 1, 10, pStereo.etenduSpeckle, &pStereo.etenduSpeckle, MAJParamStereo, &pStereo);
        AjouteGlissiere("TailleSpeckle", "Control", 3, 100, pStereo.tailleSpeckle, &pStereo.tailleSpeckle, MAJParamStereo, &pStereo);

        int alg= StereoSGBM::MODE_SGBM;
        pStereo.sgbm->setMode(alg);
    }

    for (int i = 0; i < mire.nbL; i++)
        for (int j = 0; j < mire.nbC; j++)
            for (int k=0;k<pc.size();k++)
                pc[k].pointsGrille.push_back(
                    Point3f(float(j*mire.dimCarre), float(i*mire.dimCarre), 0));
    int code =0;
    int flags= 0;
    int indexCamera=0,indImage=0;


    Mat art0, art1;
    Mat disparite;
    
    vector<vector<Point2d>> p2D;
    vector<Point2f> segment;
    for (int i = 0; i < 10;i++)
        segment.push_back(Point2f(tailleWebcam[0].width/2, tailleWebcam[0].height*i / 10.0));
    Mat equEpipolar;
    namedWindow("Webcam 0");
    namedWindow("disparite");
    setMouseCallback("Webcam 0", MesureDistance, &sDistance);
    setMouseCallback("disparite", MesureDistance, &sDistance);

    int cameraSelect=0;
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
                mtxTimeStamp[1].lock();
                pCamera[1].cmd = pCamera[1].cmd & (~MODE_REGLAGECAMERA);
                mtxTimeStamp[1].unlock();
                cameraSelect=0;
                break;
            case '1':
                mtxTimeStamp[0].lock();
                pCamera[0].cmd = pCamera[0].cmd & (~MODE_REGLAGECAMERA);
                mtxTimeStamp[0].unlock();
                mtxTimeStamp[1].lock();
                pCamera[1].cmd = pCamera[1].cmd | (MODE_REGLAGECAMERA);
                mtxTimeStamp[1].unlock();
                cameraSelect=1;
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
            switch (code) {
            case 'R':
                modeAffichage = modeAffichage| (MODE_REGLAGECAMERA);                    
                mtxTimeStamp[cameraSelect].lock();
                pCamera[cameraSelect].cmd = pCamera[cameraSelect].cmd | MODE_REGLAGECAMERA;
                mtxTimeStamp[cameraSelect].unlock();
                break;
            case 'a':
                for (int i = 0; i < v.size(); i++)
                {
                    mtxTimeStamp[i].lock();
                    if (pCamera[i].cmd & MODE_AFFICHAGE)
                        pCamera[i].cmd = pCamera[i].cmd & ~MODE_AFFICHAGE;
                    else
                        pCamera[i].cmd = pCamera[i].cmd | MODE_AFFICHAGE;
                    modeAffichage = pCamera[i].cmd;
                    mtxTimeStamp[i].unlock();
                }
                break;
            case 'b':
                frame = Mat::zeros(tailleGlobale, CV_8UC3);
                for (int i = 0; i < pc.size(); i++)
                {
                    pc[i].pointsObjets.clear();
                    pc[i].ficDonnees="";
                    pc[i].pointsCamera.clear();
                }
                sys3d.pointsCameraDroite.clear();
                sys3d.pointsCameraGauche.clear();
                sys3d.ficDonnees="";
                sys3d.pointsObjets.clear();
                break;
            case 'c':
                if (pc.size() <= indexCamera || pc[indexCamera].pointsCamera.size() == 0)
                {
                    cout << "Aucune grille pour la calibration\n";
                    break;
                }
                if (pc[indexCamera].ficDonnees.length() == 0)
                    SauverDonneesCalibration(&pc[indexCamera], NULL, NULL);
                pc[indexCamera].cameraMatrix = Mat();
                pc[indexCamera].distCoeffs = Mat();
                for (int i = 0; i<pc[indexCamera].typeCalib2D.size(); i++)
                    pc[indexCamera].rms = calibrateCamera(pc[indexCamera].pointsObjets, pc[indexCamera].pointsCamera, tailleWebcam[indexCamera], pc[indexCamera].cameraMatrix,
                        pc[indexCamera].distCoeffs, pc[indexCamera].rvecs, pc[indexCamera].tvecs, pc[indexCamera].typeCalib2D[i], TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 5000000, 1e-8));
                cout << "RMS = " << pc[indexCamera].rms << "\n";
                cout << pc[indexCamera].cameraMatrix << "\n";
                cout << pc[indexCamera].distCoeffs << "\n";

                SauverConfiguration("config.yml", mire, pc, sys3d,pStereo);
                break;
            case 'D':
                if (indexCamera != 1)
                    pc[indexCamera].pointsCamera.clear();
                indexCamera = 1;
                x = LireImages(&pCamera, &v, x);
                if (x.size() == 2 && !x[0].empty() && !x[1].empty())
                {
                    Rect dst(Point(tailleWebcam[1].width, 0), tailleWebcam[1]);
                    Mat y(tailleWebcam[1], CV_8UC3, Scalar(0, 0, 0));

                    if (AnalyseCharuco(x[indexCamera], pc[indexCamera], y, mire))
                    {
                        frame(dst) = y + frame(dst);
                        imshow("Cameras", zoom(frame, zoomAffichage,NULL));
                        if (pc[indexCamera].ficDonnees.length() != 0)
                            pc[indexCamera].ficDonnees = "";
                    }
                }
                break;
            case 'd':
                if (indexCamera != 1)
                {
                    pc[indexCamera].pointsCamera.clear();
                }
                indexCamera = 1;
                x = LireImages(&pCamera, &v, x);
                if (x.size() == 2 && !x[0].empty() && !x[1].empty())
                {
                    Rect dst(Point(tailleWebcam[1].width, 0), tailleWebcam[1]);
                    Mat y(tailleWebcam[1], CV_8UC3, Scalar(0, 0, 0));
                    if (AnalyseGrille(x[indexCamera], &pc[indexCamera], NULL, 0, y, mire))
                    {
                        frame(dst) = y + frame(dst);
                        imshow("Cameras", zoom(frame, zoomAffichage));
                        if (pc[indexCamera].ficDonnees.length() != 0)
                            pc[indexCamera].ficDonnees = "";
                    }
                }
                break;
            case 'e':
                imwrite("imgL.png", x[0]);
                imwrite("imgR.png", x[1]);
                if (!sDistance.disparite.empty())
                {
                    sDistance.disparite.convertTo(disparite, CV_32F, 1 / 16.);
                    FileStorage fs("disparite.yml", FileStorage::WRITE);
                    if (fs.isOpened())
                        fs << "Image" << disparite;
                    FileStorage fs2("disparitebrut.yml", FileStorage::WRITE);
                    if (fs2.isOpened())
                        fs2 << "Image" << sDistance.disparite;
                }
                break;
            case 'G':
                if (indexCamera != 0)
                    pc[indexCamera].pointsCamera.clear();
                indexCamera = 0;
                x = LireImages(&pCamera, &v, x);
                if (x.size() == 2)
                    if (AnalyseCharuco(x[indexCamera], pc[indexCamera],frame,mire))
                    {
                        if (pc[indexCamera].ficDonnees.length() != 0)
                            pc[indexCamera].ficDonnees = "";
                        imshow("Cameras", zoom(frame, zoomAffichage));
                    }
                break;
            case 'g':
                if (indexCamera != 0)
                    pc[indexCamera].pointsCamera.clear();
                indexCamera = 0;
                x = LireImages(&pCamera, &v, x);
                if (x.size() == 2 && !x[0].empty() && !x[1].empty())
                    if (AnalyseGrille(x[indexCamera], &pc[indexCamera], NULL, 0, frame, mire))
                    {
                        if (pc[indexCamera].ficDonnees.length() != 0)
                            pc[indexCamera].ficDonnees = "";
                        imshow("Cameras", zoom(frame, zoomAffichage));
                    }
                break;
            case 'l':
                if (modeAffichage&MODE_EPIPOLAIRE)
                    modeAffichage = modeAffichage & (~MODE_EPIPOLAIRE);
                else
                    modeAffichage = modeAffichage | (MODE_EPIPOLAIRE);

                if (!sys3d.F.empty())
                {
                    computeCorrespondEpilines(segment, 1, sys3d.F, equEpipolar);
                    cout << equEpipolar;
                }
                break;
            case 'o':
                if (!sDistance.disparite.empty())
                {
                    Mat xyz;
                    sDistance.disparite.convertTo(disparite, CV_32F, 1 / 16.);
                    reprojectImageTo3D(disparite, xyz, sys3d.Q, true);
#ifdef USE_VTK
                    VizMonde(x[0],  xyz);
#else
                    cout<<"VTK non installé";
#endif
                }
                break;
            case 'O':
                if (!sDistance.disparite.empty())
                {
                    sDistance.disparite.convertTo(disparite, CV_32F, 1 / 16.);
#ifdef USE_VTK
                    VizDisparite(x[0], disparite);
#else
                    cout << "VTK non installé";
#endif
                }
                break;
            case 's':
                if (indexCamera != 3)
                {
                    sys3d.pointsCameraGauche.clear();
                    sys3d.pointsCameraDroite.clear();
                    sys3d.pointsObjets.clear();
                    if (sys3d.ficDonnees != "")
                        sys3d.ficDonnees = "";
                }
                indexCamera = 3;
                x = LireImages(&pCamera, &v,x);
                if (x.size() == 2)
                {
                    vector<Point2f> echiquierg, echiquierd;
                    bool grilleg = findChessboardCorners(x[0], Size(mire.nbC, mire.nbL), echiquierg, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
                    bool grilled = findChessboardCorners(x[1], Size(mire.nbC, mire.nbL), echiquierd, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
                    if (grilleg && grilled)
                    {
                        Rect dst(Point(tailleWebcam[1].width, 0), tailleWebcam[1]);
                        Mat y(tailleWebcam[1], CV_8UC3, Scalar(0, 0, 0));
                        AnalyseGrille(x[0], &pc[0], &sys3d, 0, frame, mire);
                        AnalyseGrille(x[1], &pc[0], &sys3d, 1, y, mire);
                        frame(dst) = y + frame(dst);
                        imshow("Cameras", zoom(frame, zoomAffichage));
                    }
                }
                break;
            case 'S':
                if (indexCamera != 3)
                {
                    for (int i = 0; i < pc.size(); i++)
                    {
                        sys3d.pointsCameraGauche.clear();
                        sys3d.pointsCameraDroite.clear();
                        sys3d.pointsObjets.clear();
                        if (sys3d.ficDonnees != "")
                            sys3d.ficDonnees = "";
                    }
                }
                indexCamera = 3;
                x = LireImagesSynchro(&pCamera, &v);
                if (x.size() == 2)
                {
                    vector<Point2f> echiquierg, echiquierd;
                    vector< int > idsg;
                    vector< vector< Point2f > > refArucog, refus;
                    vector< int > idsd;
                    vector< vector< Point2f > > refArucod;
                    aruco::detectMarkers(x[0], mire.dictionary, refArucog, idsg, mire.detectorParams, refus);
                    aruco::refineDetectedMarkers(x[0], mire.board, refArucog, idsg, refus);
                    aruco::detectMarkers(x[1], mire.dictionary, refArucod, idsd, mire.detectorParams, refus);
                    aruco::refineDetectedMarkers(x[1], mire.board, refArucod, idsd, refus);
                    if (idsg.size() > 0 && idsd.size()>0)
                    {
                        aruco::drawDetectedMarkers(frame, refArucog, idsg);
                        Rect dst(Point(tailleWebcam[1].width, 0), tailleWebcam[1]);
                        Mat y(tailleWebcam[1], CV_8UC3, Scalar(0, 0, 0));
                        aruco::drawDetectedMarkers(y, refArucod, idsd);
                        frame(dst) = y + frame(dst);
                        imshow("Cameras", zoom(frame, zoomAffichage));
                        vector<Point3f> pReel;
                        for (int ir = 0; ir<refArucog.size(); ir++)
                        {
                            vector<int>::iterator p=find(idsd.begin(),idsd.end(), idsg[ir]);
                            if (p != idsd.end())
                            {
                                int id=(p- idsd.begin());
                                for (int jr = 0; jr<refArucog[ir].size(); jr++)
                                {
                                    echiquierg.push_back(refArucog[ir][jr]);
                                    echiquierd.push_back(refArucod[id][jr]);
                                    pReel.push_back(mire.board.get()->objPoints[idsg[ir]][jr]);
                                }
                            }
                        }
                        sys3d.pointsObjets.push_back(pReel);
                        sys3d.pointsCameraGauche.push_back(echiquierg);
                        sys3d.pointsCameraDroite.push_back(echiquierd);
                    }
                }
                break;
            case 't':
                if (!sys3d.R1.empty() && !sys3d.R2.empty())
                    if (algoStereo != 0)
                        algoStereo = 0;
                    else if (pStereo.bm)
                        algoStereo = 1;
                break;
            case 'T':
                if (!sys3d.R1.empty() && !sys3d.R2.empty())
                    if (algoStereo != 0)
                        algoStereo = 0;
                    else if (pStereo.sgbm)
                        algoStereo = 2;
                break;
            case 'u':
                typeDistorsion = (typeDistorsion +1)%4;
                cout<< typeDistorsion <<"\n";
                switch (typeDistorsion) {
                case 0:
                    map11 = Mat();
                    map12 = Mat();
                    map21 = Mat();
                    map22 = Mat();
                    break;
                case 1:
                    if (!sys3d.R1.empty())
                        initUndistortRectifyMap(sys3d.m[0], sys3d.d[0], Mat(), Mat(), tailleWebcam[0], CV_16SC2, map11, map12);
                    if (!sys3d.R2.empty())
                        initUndistortRectifyMap(sys3d.m[1], sys3d.d[1], Mat(), Mat(), tailleWebcam[1], CV_16SC2, map21, map22);
                    break;
                case 2:
                    if (!sys3d.R1.empty())
                        initUndistortRectifyMap(sys3d.m[0], Mat(), sys3d.R1, sys3d.P1, tailleWebcam[0], CV_16SC2, map11, map12);
                    if (!sys3d.R2.empty())
                        initUndistortRectifyMap(sys3d.m[1], Mat(), sys3d.R2, sys3d.P2, tailleWebcam[1], CV_16SC2, map21, map22);
                    break;
                case 3:
                    if (!sys3d.R1.empty())
                        initUndistortRectifyMap(sys3d.m[0], sys3d.d[0], sys3d.R1, sys3d.P1, tailleWebcam[0], CV_16SC2, map11, map12);
                    if (!sys3d.R2.empty())
                        initUndistortRectifyMap(sys3d.m[1], sys3d.d[1], sys3d.R2, sys3d.P2, tailleWebcam[1], CV_16SC2, map21, map22);
                    break;
                }
                break;
            case '3':
                if (sys3d.pointsCameraGauche.size() != sys3d.pointsCameraDroite.size() || sys3d.pointsCameraGauche.size() == 0)
                {
                    cout << "Pas de grille coherente pour le calibrage 3D\n";
                    break;
                }
                SauverDonneesCalibration(&pc[0], &pc[1], &sys3d);

                sys3d.m[0] = pc[0].cameraMatrix.clone();
                sys3d.m[1] = pc[1].cameraMatrix.clone();
                sys3d.d[0] = pc[0].distCoeffs.clone();
                sys3d.d[1] = pc[1].distCoeffs.clone();
                for (int i = 0; i < sys3d.typeCalib3D.size(); i++)
                {
                    sys3d.rms = stereoCalibrate(sys3d.pointsObjets, sys3d.pointsCameraGauche, sys3d.pointsCameraDroite,
                        sys3d.m[0], sys3d.d[0], sys3d.m[1], sys3d.d[1],
                        tailleWebcam[0], sys3d.R, sys3d.T, sys3d.E, sys3d.F,
                        sys3d.typeCalib3D[i],
                        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 5000000, 1e-8));
                }
                cout << "Erreur quadratique =" << sys3d.rms << endl;
                cout << ErreurDroiteEpipolaire(sys3d) << "\n";
                cout << sys3d.m[0] << "\n" << sys3d.d[0] << "\n" << sys3d.m[1] << "\n" << sys3d.d[1] << "\n";
                cout << sys3d.R << "\n" << sys3d.T << "\n" << sys3d.E << "\n" << sys3d.F << "\n";
                if (!sys3d.R.empty())
                {
                    stereoRectify(sys3d.m[0], sys3d.d[0], sys3d.m[1], sys3d.d[1],
                        tailleWebcam[0], sys3d.R, sys3d.T, sys3d.R1, sys3d.R2, sys3d.P1, sys3d.P2, sys3d.Q,
                        CALIB_ZERO_DISPARITY, -1, Size(), &sys3d.valid1, &sys3d.valid2);
                    SauverConfiguration("config.yml", mire, pc, sys3d,pStereo);
                }
                break;
            }

        }
        if (modeAffichage)
        {
            if (!algoStereo)
                x=LireImages(&pCamera, &v, x);
            else
                x= LireImagesSynchro(&pCamera, &v);
            if (x.size() == 2)
            {
                if (modeAffichage&MODE_EPIPOLAIRE && equEpipolar.rows == segment.size())
                {
                    for (int i = 0; i < equEpipolar.rows; i++)
                        {
                        circle(x[0], segment[i], 5, Scalar(255, 255, 0));
                        line(x[0], Point(0, segment[i].y), Point(x[0].cols - 1, segment[i].y), Scalar(255, 255, 0));
                        float a = equEpipolar.at<float>(i, 0);
                        float b = equEpipolar.at<float>(i, 1);
                        float c = equEpipolar.at<float>(i, 2);
                        if (a != 0)
                        {
                            float x0 = -c / a, x1 = (-b*x[1].rows - c) / a;
                            vector <Point2f> xOrig{Point2f(x0, 0), Point2f(x1, x[1].rows)};
                            line(x[1], xOrig[0], xOrig[1], Scalar(255, 255, 0));
                        }
                    //remap(x[1], x[1], map21b, map22b, INTER_LINEAR);

                    }
                }
                if (x.size() == 2 && typeDistorsion!=0 && !map11.empty() && !map21.empty() && !x[0].empty() && !x[1].empty())
                {
                    remap(x[0], x[0], map11, map12, INTER_LINEAR);
                    remap(x[1], x[1], map21, map22, INTER_LINEAR);
                }

                for (int i = 0; i < x.size(); i++)
                {
                    if (!x[i].empty())
                        imshow(format("Webcam %d", i), zoom(x[i], zoomAffichage,&sDistance));
                }
                if (modeAffichage&MODE_EPIPOLAIRE)
                {
                    Rect dst(0, 0, 0, 0);
                    for (int i = 0; i < x.size(); i++)
                    {
                        if (i == 0)
                            dst = Rect(Point(0, 0), tailleWebcam[i]);
                        else if (i % 2 == 1)
                            dst.x += tailleWebcam[i].width;
                        else
                            dst.y += tailleWebcam[i].height;
                        x[i].copyTo(frame(dst));
                    }
                    imshow("Cameras", zoom(frame, zoomAffichage));
                    if (code == 'e')
                    {
                        imwrite("epipolar.png",frame);
                    }
                }

            }
            if (algoStereo && x.size() == 2 && !x[0].empty() && !x[1].empty())
            {
                Mat disp8, disp8cc, imgL, imgD;
                if (algoStereo ==2)
                    pStereo.sgbm->compute(x[0], x[1], sDistance.disparite);
                else
                {
                    cvtColor(x[0], imgL, COLOR_BGR2GRAY);
                    cvtColor(x[1], imgD, COLOR_BGR2GRAY);
                    pStereo.bm->compute(imgL, imgD, sDistance.disparite);
                }
                sDistance.disparite.convertTo(disp8, CV_8U, 1 / 16.);
                applyColorMap(disp8, disp8cc, COLORMAP_JET);
                imshow("disparite", zoom(disp8cc, zoomAffichage));
            }
        }
    }
    while (code!=27);

    for (int i = 0; i < v.size(); i++)
    {
        mtxTimeStamp[i].lock();
    }
    stopThread=1;
    for (int i = 0; i < v.size(); i++)
    {
        mtxTimeStamp[i].unlock();
    }
    std::this_thread::sleep_for (std::chrono::seconds(2));
    th.clear();
    SauverConfiguration("config.yml", mire, pc, sys3d, pStereo);
    return 0;
}


void videoacquire(ParamCamera *pc)
{
    double aaButter[11], bbButter[11];
    aaButter[0] = -0.9996859;
    aaButter[1] = -0.9993719;
    aaButter[2] = -0.9968633;
    aaButter[3] = -0.9937365;
    aaButter[4] = -0.9690674;
    aaButter[5] = -0.9390625;
    aaButter[6] = -0.7265425;
    aaButter[7] = -0.5095254;
    aaButter[8] = -0.3249;
    aaButter[9] = -0.1584;
    aaButter[10] = -0.0;
    bbButter[0] = 0.0001571;
    bbButter[1] = 0.0003141;
    bbButter[2] = 0.0015683;
    bbButter[3] = 0.0031318;
    bbButter[4] = 0.0154663;
    bbButter[5] = 0.0304687;
    bbButter[6] = 0.1367287;
    bbButter[7] = 0.2452373;
    bbButter[8] = 0.3375;
    bbButter[9] = 0.4208;
    bbButter[10] = 0.5;
    int indFiltreMoyenne=0;

    int64 tpsInit = getTickCount();
    int64 tckPerSec = static_cast<int64> (getTickFrequency());
    int modeAffichage = 0;
    int64  tpsFrame = 0, tpsFramePre;
    int64 tpsFrameAsk;//, periodeTick = static_cast<int64> (getTickFrequency() / pc->fps);
    Mat frame,cFrame;
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
        frame = bbButter[indFiltreMoyenne] * (frame1 + frame2) - aaButter[indFiltreMoyenne] * frame;
        frame1.copyTo(frame2);
        frame.copyTo(frame1);

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

vector<VideoCapture> RechercheCamera()
{
    vector<VideoCapture> v;
    FileStorage fs("config.xml", FileStorage::READ);
    if (fs.isOpened())
    {
        FileNode n = fs["cam"];
        if (!n.empty())
        {
            FileNodeIterator it = n.begin();
            int nbCamera = 0;
            while (it != n.end() && nbCamera<NBCAMERA)
            {
                nbCamera++;
                FileNode p = *it;
                int i;
                Size s;
                (*it) >> i;
                it++;
                (*it) >> s;
                it++;
                VideoCapture vid(i);

                if (vid.isOpened() )
                {
                    vid.set(CAP_PROP_FRAME_WIDTH, s.width);
                    vid.set(CAP_PROP_FRAME_HEIGHT, s.height);
                    v.push_back(vid);
                }
                else
                    vid.release();
            }

        }
        fs.release();
    }
    else
    {

        for (size_t i = 0; i<NBCAMERA; i++)
        {
            VideoCapture video;
            video.open(static_cast<int>(i));
            if (!video.isOpened())
            {
                cout << " cannot openned camera : " << i << endl;
            }
            else
            {
                video.set(CAP_PROP_FRAME_WIDTH, 640);
                video.set(CAP_PROP_FRAME_HEIGHT, 480);
                v.push_back(video);
            }
        }
    }
    return v;
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

vector<Mat> LireImagesSynchro(vector<ParamCamera> *pc, vector<VideoCapture> *v)
{
    vector<Mat> x;
    for (int i = 0; i<v->size(); i++)
        mtxTimeStamp[i].lock();
    int64 tps;
    for (int i = 0; i < v->size(); i++)
    {
        tps = getTickCount() + (*pc)[i].tempsDelai * getTickFrequency() ;
        (*pc)[i].captureImage = 1;
        (*pc)[i].tpsCapture = tps;
        mtxTimeStamp[i].unlock();

    }
    int64 sleepTime = static_cast<int64>(25000+(tps  - getTickCount()) / getTickFrequency() * 1000000);
    std::this_thread::sleep_for(std::chrono::microseconds(sleepTime));
    for (int i = 0; i < v->size(); i++)
    {
        mtxTimeStamp[i].lock();
    }
    Rect dst(0, 0, 0, 0);
    for (int i = 0; i < v->size(); i++)
    {
        if ((*pc)[i].captureImage == 0)
        {
            x.push_back((*pc)[i].imAcq[0]);
            (*pc)[i].imAcq.pop_back();
            (*pc)[i].debAcq.pop_back();
            (*pc)[i].finAcq.pop_back();
        }
        else
            (*pc)[i].captureImage = 0;

        mtxTimeStamp[i].unlock();
    }
    if (x.size() == 1)
    {
        (*pc)[0].tempsDelai += 0.01;
        (*pc)[1].tempsDelai += 0.01;

    }
    if (x.size()!=v->size())
        cout<< "Image perdue!\n";
    return x;
}

bool ChargerConfiguration(String nomFichierConfiguration, ParamMire &mire, vector<ParamCalibration> &pc, ParamCalibration3D &sys3d, ParamAlgoStereo &pStereo)
{
    FileStorage fs(nomFichierConfiguration, FileStorage::READ);
    pc[0].index = 0;
    pc[1].index = 1;
    pc[0].indexUSB = 0;
    pc[1].indexUSB = 1;
    if (fs.isOpened())
    {
        if (!fs["EchiquierNbL"].empty())
            fs["EchiquierNbL"] >> mire.nbL;
        if (!fs["EchiquierNbC"].empty())
            fs["EchiquierNbC"]>> mire.nbC;
        if (!fs["EchiquierDimCarre"].empty())
            fs["EchiquierDimCarre"] >> mire.dimCarre;
        if (!fs["ArucoNbL"].empty())
            fs["ArucoNbL"] >> mire.nbLAruco;
        if (!fs["ArucoNbC"].empty())
            fs["ArucoNbC"] >> mire.nbCAruco;
        if (!fs["ArucoDim"].empty())
            fs["ArucoDim"] >> mire.dimAruco;
        if (!fs["ArucoSep"].empty())
            fs["ArucoSep"] >> mire.sepAruco;
        if (!fs["ArucoDict"].empty())
            fs["ArucoDict"] >> mire.dict;
        if (!fs["Cam0index"].empty())
            fs["Cam0index"] >> pc[0].indexUSB;
        if (!fs["Cam0Size"].empty())
            fs["Cam0Size"] >> pc[0].tailleImage;
        if (!fs["Cam1Size"].empty())
            fs["Cam1Size"] >> pc[1].tailleImage;
        if (!fs["Cam1index"].empty())
            fs["Cam1index"] >> pc[1].indexUSB;
        if (!fs["typeCalib1"].empty())
            fs["typeCalib1"] >> pc[1].typeCalib2D;
        if (!fs["typeCalib0"].empty())
            fs["typeCalib0"] >> pc[0].typeCalib2D;
        if (!fs["typeCalib3d"].empty())
            fs["typeCalib3d"] >> sys3d.typeCalib3D;
        if (!fs["cameraMatrice0"].empty())
            fs["cameraMatrice0"] >> pc[0].cameraMatrix;
        if (!fs["cameraDistorsion0"].empty())
            fs["cameraDistorsion0"] >> pc[0].distCoeffs;
        if (!fs["cameraMatrice1"].empty())
            fs["cameraMatrice1"] >> pc[1].cameraMatrix;
        if (!fs["cameraDistorsion1"].empty())
            fs["cameraDistorsion1"] >> pc[1].distCoeffs;
        if (!fs["ficDonnee0"].empty())
        {
            fs["ficDonnee0"] >> pc[0].ficDonnees;
            if (pc[0].ficDonnees.length()>0)
                ChargerDonneesCalibration(pc[0].ficDonnees, &pc[0], NULL);

        }
        if (!fs["ficDonnee1"].empty() )
        {
            fs["ficDonnee1"] >> pc[1].ficDonnees;
            if (pc[1].ficDonnees.length()>0)
                ChargerDonneesCalibration(pc[1].ficDonnees, &pc[1], NULL);
        }
        if (!fs["ficDonnee3d"].empty())
        {
            fs["ficDonnee3d"] >> sys3d.ficDonnees;
            if (sys3d.ficDonnees.length()>0)
                ChargerDonneesCalibration(sys3d.ficDonnees, NULL, &sys3d);
        }
        if (!fs["R"].empty())
        {
            fs["R"] >> sys3d.R;
            fs["T"]>> sys3d.T;
            fs["R1"]>>sys3d.R1;
            fs["R2"]>>sys3d.R2;
            fs["P1"]>>sys3d.P1;
            fs["P2"]>>sys3d.P2;
            fs["Q"]>>sys3d.Q;
            fs["F"]>>sys3d.F;
            fs["E"]>>sys3d.E;
            fs["rect1"]>>sys3d.valid1;
            fs["rect2"]>>sys3d.valid2;
            sys3d.m.resize(2);
            sys3d.d.resize(2);
            fs["M0"]>>sys3d.m[0];
            fs["D0"]>>sys3d.d[0];
            fs["M1"]>>sys3d.m[1];
            fs["D1"]>>sys3d.d[1];
        }
        if (!fs["typeAlgoSGBM"].empty())
        {
            fs["typeAlgoSGBM"]>> pStereo.typeAlgoSGBM;
            fs["preFilterType"] >> pStereo.typePreFiltre;
            fs["preFilterCap"] >> pStereo.preFilterCap;
            fs["blockSize"] >> pStereo.tailleBloc;
            fs["numDisparities"] >> pStereo.nbDisparite;
            fs["uniquenessRatio"] >> pStereo.unicite;
            fs["speckleRange"] >> pStereo.etenduSpeckle;
            fs["speckleSize"] >> pStereo.tailleSpeckle;
        }
        return true;
    }
    return false;
}


void SauverConfiguration(String nomFichierConfiguration,ParamMire mire,vector<ParamCalibration> pc, ParamCalibration3D &sys3d, ParamAlgoStereo &pStereo)
{
    if (nomFichierConfiguration == "config.yml")
    {
        time_t rawtime;
        struct tm * timeinfo;
        char buffer[256];

        time(&rawtime);
        timeinfo = localtime(&rawtime);

        strftime(buffer, 256, "config%Y%m%d%H%M%S.yml", timeinfo);
        rename(nomFichierConfiguration.c_str(), buffer);
    }
    FileStorage fs(nomFichierConfiguration, FileStorage::WRITE);

    if (fs.isOpened())
    {
        time_t t1;
        time(&t1);
        struct tm *t2 = localtime(&t1);
        char tmp[1024];
        strftime(tmp, sizeof(tmp) - 1, "%c", t2);

        fs << "date" << tmp;
        fs << "EchiquierNbL"<<mire.nbL;
        fs << "EchiquierNbC"<<mire.nbC;
        fs << "EchiquierDimCarre"<<mire.dimCarre;
        fs << "ArucoNbL"<<mire.nbLAruco;
        fs << "ArucoNbC" << mire.nbCAruco;
        fs << "ArucoDim" << mire.dimAruco;
        fs << "ArucoSep"<< mire.sepAruco;
        fs << "ArucoDict" << mire.dict;
        fs << "typeCalib0" << pc[0].typeCalib2D;
        fs << "Cam0index" << pc[0].indexUSB;
        fs << "Cam0Size" << pc[0].tailleImage;
        fs << "typeCalib1" << pc[1].typeCalib2D;
        fs << "Cam1index" << pc[1].indexUSB;
        fs << "Cam1Size" << pc[1].tailleImage;
        fs << "cameraMatrice0" << pc[0].cameraMatrix;
        fs << "cameraDistorsion0" << pc[0].distCoeffs;
        fs << "cameraMatrice1" << pc[1].cameraMatrix;
        fs << "cameraDistorsion1" << pc[1].distCoeffs;
        fs << "ficDonnee0" << pc[0].ficDonnees;
        fs << "ficDonnee1" << pc[1].ficDonnees;
        fs << "ficDonnee3d" << sys3d.ficDonnees;
        fs << "typeCalib3d" << sys3d.typeCalib3D;
        fs << "R" << sys3d.R << "T" << sys3d.T << "R1" << sys3d.R1 << "R2" << sys3d.R2 << "P1" << sys3d.P1 << "P2" << sys3d.P2;
        fs << "Q" << sys3d.Q << "F" << sys3d.F << "E" << sys3d.E << "rect1" << sys3d.valid1 << "rect2" << sys3d.valid2;
        if (sys3d.m.size() >= 2)
        {
            fs << "M0" << sys3d.m[0] << "D0" << sys3d.d[0];
            fs << "M1" << sys3d.m[1] << "D1" << sys3d.d[1];

        }
        fs << "typeAlgoSGBM" << pStereo.typeAlgoSGBM;
        fs << "preFilterType" << pStereo.typePreFiltre;
        fs << "preFilterCap" << pStereo.preFilterCap;
        fs << "blockSize" << pStereo.tailleBloc;
        fs << "numDisparities" << pStereo.nbDisparite;
        fs << "uniquenessRatio" << pStereo.unicite;
        fs << "speckleRange" << pStereo.etenduSpeckle;
        fs << "speckleSize" << pStereo.tailleSpeckle;

    }
}


vector<Mat> LireImages(vector<ParamCamera> *pc, vector<VideoCapture> *v,vector<Mat> &x)
{
    vector<Mat> xx;
    if (x.size()!=v->size())
        xx.resize(v->size());
    else
        xx=x;
    for (int i = 0; i < v->size(); i++)
    {
        mtxTimeStamp[i].lock();
        (*pc)[i].derniereImage.copyTo(xx[i]);
        mtxTimeStamp[i].unlock();
    }
    return xx;
}

#ifdef USE_VTK

void VizMonde(Mat img, Mat xyz)
{
    vector<Point3d> points;
    vector<Vec3b> couleur;
    for (int i = 0; i < xyz.rows; i++)
    {
        Vec3f *pLig = (Vec3f*)(xyz.ptr(i));
        for (int j = 0; j < xyz.cols ; j++, pLig++)
        {
            if (pLig[0][2] < 10000 )
            {
                Vec3d p1(pLig[0][0], pLig[0][1], pLig[0][2]);
                points.push_back(p1);
                couleur.push_back(img.at<Vec3b>(i, j));
            }
        }
    }
    viz::Viz3d fen3D("Monde");
    viz::WCloud nuage(points,  couleur);
    fen3D.showWidget("I3d", nuage);
    fen3D.spin();
}


void VizDisparite(Mat img, Mat disp)
{
    vector<Point3d> points;
    vector<int> polygone;
    vector<Vec3b> couleur;
    int nbPoint = 0;
    for (int i = 1; i < img.rows - 1; i++)
    {
        float *d = disp.ptr<float>(i) + 1;
        for (int j = 1; j < img.cols - 1; j++, d++)
        {
            float disparite= *d;
            if (disparite<0)
                disparite =10000;
            Vec3d p1(j,i, *d);
            Vec3d p2(j-1,i , *d);
            Vec3d p3(j,i-1, *d);
            Vec3d p4(j-1,i-1, *d);
            points.push_back(p1);
            points.push_back(p2);
            points.push_back(p3);
            points.push_back(p4);
            if (*d >= 0)
            {
                couleur.push_back(img.at<Vec3b>(i, j));
                couleur.push_back(img.at<Vec3b>(i, j));
                couleur.push_back(img.at<Vec3b>(i, j));
                couleur.push_back(img.at<Vec3b>(i, j));
            }
            else
            {
                couleur.push_back(Vec3b(0,0,0));
                couleur.push_back(Vec3b(0, 0, 0));
                couleur.push_back(Vec3b(0, 0, 0));
                couleur.push_back(Vec3b(0, 0, 0));
            }
            polygone.push_back(4);
            polygone.push_back(nbPoint);
            polygone.push_back(nbPoint + 2);
            polygone.push_back(nbPoint + 1);
            polygone.push_back(nbPoint + 3);
            nbPoint += 4;
        }
    }
    viz::Viz3d fen3D("Disparite 3D");
    viz::WMesh reseauFacette(points, polygone, couleur);
    fen3D.showWidget("I3d", reseauFacette);
    fen3D.spin();
}
#endif

void MesureDistance(int event, int x, int y, int flags, void * userData)
{
    SuiviDistance *sDistance=(SuiviDistance*)userData;
    if (sDistance->disparite.empty())
        return;
    if (event == EVENT_FLAG_LBUTTON)
    {
        Point pImg(x / sDistance->zoomAffichage, y / sDistance->zoomAffichage);
        if (sDistance->disparite.at<short>(pImg) <=-1)
            return;
        Point3d p(x / sDistance->zoomAffichage, y/ sDistance->zoomAffichage, sDistance->disparite.at<short>(pImg));
        p.z = p.z/16;
        sDistance->p.push_back(p);
        Mat ptDisparite(sDistance->p), ptXYZ;
        sDistance->m.release();
        perspectiveTransform(ptDisparite, sDistance->m, sDistance->pStereo->Q);
        cout << "\n ++++++++++\n";
        cout << ptDisparite;
        cout << "\n ";
        for (int i=0;i<sDistance->m.rows;i++)
            cout << ptDisparite.at<Vec3d>(i)<<" = "<<sDistance->m.at<Vec3d>(i)<<" --> "<<norm(sDistance->m.at<Vec3d>(i))<<"\n";
    }
    if (event == EVENT_FLAG_RBUTTON)
    {
        sDistance->p.clear();
    }
}

Mat zoom(Mat x, float w,SuiviDistance *s)
{
    if (s && !s->disparite.empty() && s->p.size() > 0)
    {

    }
    if (w!=1)
    {
        Mat y;
        resize(x,y,Size(),w,w);
        return y;
    }
    return x;
 }

void AjouteGlissiere(String nomGlissiere, String nomFenetre, int minGlissiere, int maxGlissiere, int valeurDefaut, int *valGlissiere, void(*f)(int, void *), void *r)
{
    createTrackbar(nomGlissiere, nomFenetre, valGlissiere, 1, f, r);
    setTrackbarMin(nomGlissiere, nomFenetre, minGlissiere);
    setTrackbarMax(nomGlissiere, nomFenetre, maxGlissiere);
    setTrackbarPos(nomGlissiere, nomFenetre, valeurDefaut);
}

void MAJParamStereo(int x, void * r)
{
    ParamAlgoStereo *pStereo= (ParamAlgoStereo *)r;

    pStereo->bm->setBlockSize(2*pStereo->tailleBloc+1);
    pStereo->sgbm->setBlockSize(2 * pStereo->tailleBloc + 1);
    pStereo->bm->setNumDisparities(16*pStereo->nbDisparite);
    pStereo->sgbm->setNumDisparities(16*pStereo->nbDisparite);
    pStereo->bm->setUniquenessRatio(pStereo->unicite);
    pStereo->sgbm->setUniquenessRatio(pStereo->unicite);
    pStereo->bm->setSpeckleWindowSize(pStereo->tailleSpeckle);
    pStereo->sgbm->setSpeckleWindowSize(pStereo->tailleSpeckle);
    pStereo->bm->setSpeckleRange(pStereo->etenduSpeckle);
    pStereo->sgbm->setSpeckleRange(pStereo->etenduSpeckle);
}


double ErreurDroiteEpipolaire(ParamCalibration3D sys3d)
{
    double err = 0;
    int nbPoints = 0;
    vector<Vec3f> lines[2];
    for (int i = 0; i < sys3d.pointsCameraGauche.size(); i++)
    {
        int nbPt = (int)sys3d.pointsCameraGauche[i].size();
        Mat ptImg[2];
        undistortPoints(sys3d.pointsCameraGauche[i], ptImg[0], sys3d.m[0], sys3d.d[0], Mat(), sys3d.m[0]);
        undistortPoints(sys3d.pointsCameraDroite[i], ptImg[1], sys3d.m[1], sys3d.d[1], Mat(), sys3d.m[1]);
        computeCorrespondEpilines(ptImg[0], 1, sys3d.F, lines[0]);
        computeCorrespondEpilines(ptImg[1], 2, sys3d.F, lines[1]);
        for (int j = 0; j < nbPt; j++)
        {
            double errij1 = fabs(ptImg[0].at<Vec2f>(0,j)[0]*lines[1][j][0] +
                ptImg[0].at<Vec2f>(0, j)[1] *lines[1][j][1] + lines[1][j][2]);
            double errij2 = fabs(ptImg[1].at<Vec2f>(0, j)[0] *lines[0][j][0] +
                ptImg[1].at<Vec2f>(0, j)[1]*lines[0][j][1] + lines[0][j][2]);
            err += errij1 + errij2;
        }
        nbPoints += nbPt;
    }
    return err/ nbPoints/2;
}

bool AnalyseCharuco(Mat x, ParamCalibration &pc,Mat frame,ParamMire &mire)
{
    
    vector< int > ids;
    vector< vector< Point2f > > refAruco, refus;
    vector<Point2f> echiquier;
    aruco::detectMarkers(x, mire.dictionary, refAruco, ids, mire.detectorParams);
 
    aruco::refineDetectedMarkers(x, mire.board, refAruco, ids, refus);
    if (ids.size() > 0)
    {
        Mat currentCharucoCorners, currentCharucoIds;
        aruco::interpolateCornersCharuco(refAruco, ids, x, mire.gridboard, currentCharucoCorners,
            currentCharucoIds);
        aruco::drawDetectedCornersCharuco(frame, currentCharucoCorners, currentCharucoIds);
        vector<Point3f> pReel;
        for (int ir = 0; ir<refAruco.size(); ir++)
            for (int jr = 0; jr<refAruco[ir].size(); jr++)
            {
                echiquier.push_back(refAruco[ir][jr]);
                pReel.push_back(mire.board.get()->objPoints[ids[ir]][jr]);
            }

        for (int ir = 0; ir < currentCharucoIds.rows; ir++)
        {
            int index= currentCharucoIds.at<int>(ir, 0);
            echiquier.push_back(currentCharucoCorners.at<Point2f>(ir,0));
            pReel.push_back(mire.gridboard->chessboardCorners[index]);

        }
        pc.pointsObjets.push_back(pReel);
        pc.pointsCamera.push_back(echiquier);
    }
    else
        return false;
    return true;
}

bool AnalyseGrille(Mat x, ParamCalibration *pc, ParamCalibration3D *sys3d,int index, Mat frame, ParamMire &mire)
{
    vector<Point2f> echiquier;
    bool grille = findChessboardCorners(x, Size(mire.nbC, mire.nbL), echiquier, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
    if (grille)
    {
        Mat imGris;
        cvtColor(x, imGris, COLOR_BGR2GRAY);
        cornerSubPix(imGris, echiquier, Size(5, 5), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.01));
        drawChessboardCorners(frame, Size(mire.nbC, mire.nbL), Mat(echiquier), false);
        if (sys3d==NULL && pc)
        {
            pc->pointsCamera.push_back(echiquier);
            pc->pointsObjets.push_back(pc->pointsGrille);
        }
        if (sys3d)
        {
            if (index==0)
            {
                sys3d->pointsCameraGauche.push_back(echiquier);
                sys3d->pointsObjets.push_back(pc->pointsGrille);
            }
            else
                sys3d->pointsCameraDroite.push_back(echiquier);
        }

    }
    else
        return false;
    return true;
}

void ChargerDonneesCalibration(String nomFichier, ParamCalibration *pc, ParamCalibration3D *sys3d)
{
    FileStorage fs(nomFichier, FileStorage::READ);
    if (!fs.isOpened())
    {
        return;
    }
    if (pc)
    {
        if (!fs["nbGrille"].empty())
            fs["nbGrille"] >> pc->nbImageGrille;
        if (!fs["Grille"].empty())
            fs["Grille"] >> pc->pointsCamera;
        if (!fs["Objet"].empty())
            fs["Objet"] >> pc->pointsObjets;
    }
    else if( sys3d)
    {
        if (!fs["GrilleG"].empty())
            fs["GrilleG"] >> sys3d->pointsCameraGauche;
        if (!fs["GrilleD"].empty())
            fs["GrilleD"] >> sys3d->pointsCameraDroite;
        if (!fs["Objet"].empty())
            fs["Objet"] >> sys3d->pointsObjets;

    }
    fs.release();
}


void SauverDonneesCalibration(ParamCalibration *pc1, ParamCalibration *pc2,ParamCalibration3D *sys3d)
{
    if (!sys3d)
    {
        ParamCalibration *pc;

        if (pc1)
            pc =pc1;
        else
            pc =pc2;
        int nbPts=0;
        for (int i=0;i<pc->pointsCamera.size();i++)
            nbPts+= pc->pointsCamera[i].size();
        pc->ficDonnees= format("Echiquier%d_%d.yml", pc->indexUSB, getTickCount());
        FileStorage fEchiquier(pc->ficDonnees, FileStorage::WRITE);
        fEchiquier << "nbGrille" << (int)pc->pointsCamera.size();
        fEchiquier << "nbPoints" << (int)nbPts;
        fEchiquier << "Grille" << pc->pointsCamera;
        fEchiquier << "Objet" << pc->pointsObjets;
        fEchiquier.release();

    }
    else
    {
        sys3d->ficDonnees = format("EchiquierStereo_%d.yml", getTickCount());
        FileStorage fEchiquier(sys3d->ficDonnees, FileStorage::WRITE);
        int nbPts = 0;
        for (int i = 0; i<pc1->pointsCamera.size(); i++)
            nbPts += pc1->pointsCamera[i].size();
        fEchiquier << "nbGrille" << (int)pc1->pointsCamera.size();
        fEchiquier << "nbPoints" << (int)nbPts;
        fEchiquier << "GrilleG" << sys3d->pointsCameraGauche;
        fEchiquier << "GrilleD" << sys3d->pointsCameraDroite;
        fEchiquier << "Objet" << sys3d->pointsObjets;
        fEchiquier.release();
    }

}

