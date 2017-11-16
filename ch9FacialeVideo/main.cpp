#include <opencv2/opencv.hpp> 
#include <opencv2/face.hpp>
using namespace std;
using namespace cv;


Mat ExtractionVisage(Mat img, Rect r, CascadeClassifier &yeux,vector<Rect> &zoneOeil);
const String keys =
"{Aide h usage ? help  |     | Afficher ce message   }"
"{@arg1                |     | nom du fichier video }"
"{d                    |     | chemin d'opencv }";

int main(int argc, char **argv)
{   
    CommandLineParser parser(argc, argv, keys);
    String nomDossierData;
    String nomFichierVideo= parser.get<String>(0);
    CascadeClassifier faceCascade;
    CascadeClassifier eyesCascade;
    String window_name = "Capture - Face detection";

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    if (parser.has("d"))
    {
        nomDossierData = parser.get<String>("d");
    }
    else
    {
        cout << " Nom du dossier absent\n";
        exit(-1);
    }

    String nomCascadeVisage = nomDossierData + "/opencv/data/lbpcascades/lbpcascade_frontalface_improved.xml";
    String nomCascadeYeux = nomDossierData + "/opencv/data/haarcascades/haarcascade_eye.xml";
    if (!faceCascade.load(nomCascadeVisage))
    { 
        cout << "Classifieur de detection de visage non trouve\n";
        return -1; 
    }
    if (!eyesCascade.load(nomCascadeYeux))
    { 
        cout<<"Classifieur de detection des yeux non trouve\n";
        return -1;
    }

    Mat labelVisagesVideo(0, 1, CV_32SC1);
    vector<Mat> visagesVideo;
    Mat labelVisagesVideoTest(0, 1, CV_32SC1);
    vector<Mat> visagesVideoTest;
    VideoCapture video(nomFichierVideo);

    if (!video.isOpened())
    {
        cout << "Video introuvable";
        return 0;
    }
    Mat frameVideo;
    video>>frameVideo;
    int fourcc=video.get(CAP_PROP_FOURCC);
    string fourcc_str = format("%c%c%c%c", fourcc & 255, (fourcc >> 8) & 255, (fourcc >> 16) & 255, (fourcc >> 24) & 255);
    cout << "CAP_PROP_FOURCC: " << fourcc_str << endl;
    video.set(CAP_PROP_POS_FRAMES, 0);
    Mat_<float> posVisagesRef(0, 2);
    vector<int> dureeVisageRef;
    vector<int> labelVisageRef;
    cv::flann::Index indexPPV;
    int histSize[] = { 32 };
    float etenduGris[] = { 0, 256 };
    const float* etenduDesCanaux[] = { etenduGris };
    int canaux[] = { 0 };
    Mat histRef;

    Ptr<face::LBPHFaceRecognizer> modelLBPHFVideo = Algorithm::load<face::LBPHFaceRecognizer>("myVideoFace.yml");
    bool modelReady=true;
    VideoWriter resultat;
    if (modelLBPHFVideo==NULL)
    {
        modelReady = false;
        modelLBPHFVideo =  face::LBPHFaceRecognizer::create(1, 8, 8, 8);
        resultat.open("Identification.avi", VideoWriter::fourcc('H', '2', '6', '4'), 24, Size(frameVideo.cols, frameVideo.rows));
    }
    else
        resultat.open("Recensement.avi", VideoWriter::fourcc('H', '2', '6', '4'), 24, Size(frameVideo.cols, frameVideo.rows));
    int indFrame = -1;
    int indVisage = 0;
    int nbEchec = 0;
    do
    {
        indFrame++;
        vector<Rect> zoneVisage;
        vector<Rect> zoneOeil;
        video>> frameVideo;
        if (!frameVideo.empty())
        {
            Mat frame;
            cvtColor(frameVideo,frame,CV_BGR2GRAY);
            faceCascade.detectMultiScale(frame, zoneVisage, 1.1, 4, 0 , Size(30, 30));
            if (modelReady)
            {
                for (size_t i = 0; i < zoneVisage.size(); i++)
                {
                    Rect rRect(zoneVisage[i].x, zoneVisage[i].y, zoneVisage[i].width, zoneVisage[i].height);
                    int label=-1;
                    double dist;
                    Mat img2 = ExtractionVisage(frame, rRect, eyesCascade,zoneOeil);
                    modelLBPHFVideo->predict(img2,label,dist);

                    Point center(zoneVisage[i].x + zoneVisage[i].width / 2, zoneVisage[i].y + zoneVisage[i].height / 2);              
                    if (label==-1)
                        ellipse(frameVideo, center, Size(zoneVisage[i].width / 2, zoneVisage[i].height / 2), 0, 0, 360, Scalar(0, 0, 255), 4, 8, 0);
                    else
                    {
                        ellipse(frameVideo, center, Size(zoneVisage[i].width / 2, zoneVisage[i].height / 2), 0, 0, 360, Scalar(0, 255, 0), 4, 8, 0);
                        putText(frameVideo,format("%d",label),center, FONT_HERSHEY_SCRIPT_SIMPLEX,1.5,Scalar(0,255,0),2);
                    }
                }
            }
            else
            {
                if (posVisagesRef.rows == 0) // rien � comparer
                {
                    calcHist(&frameVideo, 1, canaux, Mat(), histRef, 1, histSize, etenduDesCanaux,
                        true, false);
                    for (size_t i = 0; i < zoneVisage.size(); i++)
                    {
                        cv::Mat pVisage = (cv::Mat_<float>(1, 2) << zoneVisage[i].x, zoneVisage[i].y);
                        posVisagesRef.push_back(pVisage);
                        dureeVisageRef.push_back(1);
                        labelVisageRef.push_back(indVisage++);
                    }
                    if (zoneVisage.size() != 0)
                        indexPPV.build(posVisagesRef, cv::flann::KDTreeIndexParams(1));
                }
                else
                {
                    Mat_<float> posVisagesNew;
                    vector<int> dureeVisagesNew;
                    vector<int> labelVisageNew;
                    Mat hist;
                    calcHist(&frameVideo, 1, canaux, Mat(), hist, 1, histSize, etenduDesCanaux,
                        true, false);
                    double x = compareHist(hist, histRef, HISTCMP_CORREL);
                    histRef = hist;
                    for (size_t i = 0; i < zoneVisage.size() && fabs(x)>0.7; i++)
                    {
                        cv::Mat pVisage = (cv::Mat_<float>(1, 2) << zoneVisage[i].x, zoneVisage[i].y);
                        Mat indexVisage, depVisage;
                        int nbVoisins = indexPPV.radiusSearch(pVisage, indexVisage, depVisage, 100, 1);
                        if (nbVoisins == 1 )
                        {
                            cout << depVisage.at<float>(0, 0)<<"\n";
                            posVisagesNew.push_back(pVisage);
                            int index = labelVisageRef[indexVisage.at<int>(0, 0)];
                            dureeVisagesNew.push_back(dureeVisageRef[indexVisage.at<int>(0, 0)]+1);
                            labelVisageNew.push_back(index);
                            nbEchec = 0;
                            Rect rRect = zoneVisage[i];
                            if (dureeVisagesNew.front() % 9 == 7)
                            {
                                Mat img2 = ExtractionVisage(frame, rRect, eyesCascade,zoneOeil);
                                imshow(format("visage%d", index), img2);
                                labelVisagesVideo.push_back(index);
                                visagesVideo.push_back(img2);
                            }
                            else if (dureeVisagesNew.front() % 9 == 2)
                            {
                                Mat img2 = ExtractionVisage(frame, rRect, eyesCascade,zoneOeil);
                                labelVisagesVideoTest.push_back(index);
                                visagesVideoTest.push_back(img2);
                            }
                        }
                        else
                        {
                            posVisagesNew.push_back(pVisage);
                            dureeVisagesNew.push_back(1);
                            labelVisageNew.push_back(indVisage++);
                        }
                    }
                    if (dureeVisagesNew.size() != 0 && fabs(x)>0.7)
                    {
                        posVisagesRef = posVisagesNew;
                        dureeVisageRef = dureeVisagesNew;
                        labelVisageRef = labelVisageNew;
                        indexPPV.build(posVisagesRef, cv::flann::KDTreeIndexParams(1));
                    }
                    else
                    {
                        nbEchec++;
                        if (nbEchec > 5 || fabs(x)<0.7)
                        {// Fin du plan- insertion visage dans la base
                            dureeVisageRef.clear();
                            posVisagesRef.release();
                            labelVisageRef.clear();
                            if (modelLBPHFVideo->getLabels().empty() && labelVisagesVideo.rows>=10) // initialisation de la base
                            {
                                modelLBPHFVideo->train(visagesVideo, labelVisagesVideo);
                                double distVisage = 0.0;
                                double distMean = 0.0;
                                double distMean2 = 0.0;
                                int label = -1;
                                for (int k = 0; k < labelVisagesVideoTest.rows; k++)
                                {
                                    modelLBPHFVideo->predict(visagesVideoTest[k], label, distVisage);
                                    distMean += distVisage;
                                    distMean2 += distVisage*distVisage;
                                }
                                distMean = distMean / labelVisagesVideoTest.rows;
                                distMean2 = distMean2 / labelVisagesVideoTest.rows;
                                for (int k = 0; k < labelVisagesVideo.rows; k++)
                                {
                                    imwrite(format("visage%d_%d.jpg", labelVisagesVideo.at<int>(k, 0), indFrame+k), visagesVideo[k]);
                                }
                                modelLBPHFVideo->setThreshold(distMean + 0.15*sqrt(distMean2 - distMean*distMean));
                                cout << "\n" << modelLBPHFVideo->getThreshold() << "\n";
                            } 
                            else if (!modelLBPHFVideo->getLabels().empty()) // MAJ de la base
                            {
                                map<int, int> imgDsBase; //label ancien,label nouveau
                                map<int, double> imgDistBase;// distance image base
                                map<int, int> listLabel;// liste des labels � �tudier

                                for (int k = 0; k < labelVisagesVideo.rows; k++)
                                {
                                    double distVisage = 0.0;
                                    int label = -1;
                                    int index = labelVisagesVideo.at<int>(k, 0);
                                    if (listLabel.find(index) == listLabel.end())
                                        listLabel.insert(pair<int, int>(index, 1));
                                    else
                                        listLabel.find(index)->second++;
                                    modelLBPHFVideo->predict(visagesVideo[k], label, distVisage);
                                    map<int, double>::iterator ited = imgDistBase.find(label);
                                    map<int, int>::iterator ite = imgDsBase.find(label);
                                    if (ite != imgDsBase.end())
                                    {
                                        if (ited->second > distVisage)
                                        {
                                            ite->second = labelVisagesVideo.at<int>(k, 0);
                                            ited->second = distVisage;
                                        }
                                    }
                                    else if (label!=-1)
                                    {
                                        imgDsBase.insert(pair<int, int>(label, labelVisagesVideo.at<int>(k, 0)));
                                        imgDistBase.insert(pair<int, double>(label, distVisage));
                                    }
                                }
                                map<int, int>::iterator ite = imgDsBase.begin();
                                for (; ite != imgDsBase.end(); ite++)
                                {
                                    int index = ite->second;
                                    map<int, double>::iterator ited = imgDistBase.find(ite->first);
                                    cout << index<< " -> "<< ite->first <<"("<< ited->second<<")\t";
                                    for (int k = 0; k<visagesVideo.size(); k++)
                                        if (index == labelVisagesVideo.at<int>(k, 0))
                                        {
                                            labelVisagesVideo.at<int>(k, 0) = ite->first;
                                        }

                                }
                                if (visagesVideo.size() > 0)
                                {
                                    for (int k = 0; k<visagesVideo.size(); k++)
                                        imwrite(format("visage%d_%d.jpg", labelVisagesVideo.at<int>(k, 0), indFrame+k), visagesVideo[k]);

                                    modelLBPHFVideo->update(visagesVideo, labelVisagesVideo);
                                }
                                visagesVideo.clear();
                                labelVisagesVideo.release();
                                visagesVideoTest.clear();
                                labelVisagesVideoTest.release();

                            }
                        }
                    }
                }
                for (size_t i = 0; i < zoneVisage.size(); i++)
                {
                    Point center(zoneVisage[i].x + zoneVisage[i].width / 2, zoneVisage[i].y + zoneVisage[i].height / 2);
                    ellipse(frameVideo, center, Size(zoneVisage[i].width / 2, zoneVisage[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
                    for (int j = 0; j < zoneOeil.size(); j++)
                    {
                        Point center(zoneOeil[j].x + zoneOeil[j].width / 2, zoneOeil[i].y + zoneOeil[i].height / 2);
                        ellipse(frameVideo, center+zoneVisage[i].tl(), Size(zoneOeil[i].width / 2, zoneOeil[i].height / 2), 0, 0, 360, Scalar(0, 255, 255), 4, 8, 0);

                    }
                }

            }
            imshow("video",frameVideo);
            resultat<<frameVideo;
            int code = waitKey(10);
            if (code==32)
                waitKey();
        }
        //frameVideo= Mat();
    }
    while (!frameVideo.empty());
    resultat.release();
    if (!modelReady)
    {
        modelLBPHFVideo->save("myVideoFace.yml");
        FileStorage fs("myVideoFace.yml",FileStorage::APPEND);
        fs.release();
        cout << modelLBPHFVideo->getLabels()<<"\n";
    }
    return 0;
}

Mat ExtractionVisage(Mat frame, Rect r, CascadeClassifier &yeux, vector<Rect> &zoneOeil)
{
    Size vignette(150, 150);
    Mat img=frame(r).clone();
    yeux.detectMultiScale(img, zoneOeil, 1.1, 4, 0, Size(30, 30));
    if (zoneOeil.size() == 2)
    {
        Point centreOeilGauche;
        Point centreOeilDroit;
        if (norm(zoneOeil[0].tl())<norm(zoneOeil[1].tl()))
        {
            centreOeilGauche = ((zoneOeil[0].tl() + zoneOeil[0].br()) / 2);
            centreOeilDroit = ((zoneOeil[1].tl() + zoneOeil[1].br()) / 2);
        }
        else
        {
            centreOeilDroit = ((zoneOeil[0].tl() + zoneOeil[0].br()) / 2);
            centreOeilGauche = ((zoneOeil[1].tl() + zoneOeil[1].br()) / 2);
        }
        double angle = atan2(centreOeilDroit.y - centreOeilGauche.y, centreOeilDroit.x - centreOeilGauche.x) * 180 / CV_PI;
        Mat rot2d = getRotationMatrix2D((centreOeilGauche + centreOeilDroit) / 2, angle, 1);
        vector<Point3d> p = { Point3d(0,0,1),Point3d(0,img.rows,1),Point3d(img.cols,img.rows,1),Point3d(img.cols,0,1) };
        Mat pDst = (rot2d*Mat(Point3d(0, 0, 1))).t();
        Point2d pMin(pDst.at<Vec2d>(0, 0)), pMax(pDst.at<Vec2d>(0, 0));
        for (int i = 1; i<4; i++)
        {
            pDst = (rot2d*Mat(p[i])).t();
            if (pDst.at<Vec2d>(0, 0)[0]<pMin.x)
                pMin.x = pDst.at<Vec2d>(0, 0)[0];
            if (pDst.at<Vec2d>(0, 0)[1]<pMin.y)
                pMin.y = pDst.at<Vec2d>(0, 0)[1];
            if (pDst.at<Vec2d>(0, 0)[0]>pMax.x)
                pMax.x = pDst.at<Vec2d>(0, 0)[0];
            if (pDst.at<Vec2d>(0, 0)[1]>pMax.y)
                pMax.y = pDst.at<Vec2d>(0, 0)[1];
        }

        rot2d.at<double>(0, 2) -= pMin.x;
        rot2d.at<double>(1, 2) -= pMin.y;
        Mat img2;
        float fy = vignette.height / float(pMax.y - pMin.y), fx = vignette.width / float(pMax.x - pMin.x);
        warpAffine(img, img2, rot2d, Size(pMax.x - pMin.x,pMax.y - pMin.y ), INTER_LINEAR, BORDER_CONSTANT, Scalar(128));
        resize(img2, img2, Size(), min(fx, fy), min(fx, fy));
        Mat img3(vignette, CV_8UC1, Scalar(128));
        img2.copyTo(img3(Rect(0, 0, img2.cols, img2.rows)));
        return img3;
    }
    float fy = vignette.height / float(img.rows), fx = vignette.width / float(img.cols);
    resize(img, img, Size(), min(fx, fy), min(fx, fy));
    return img;

}

