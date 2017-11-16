#include <opencv2/opencv.hpp> 
#include <iostream>

#ifdef HAVE_OPENCV_XIMGPROC
#include <opencv2/ximgproc.hpp>
#endif


using namespace std;
using namespace cv;
using namespace cv::ml;

enum {
    DESC_MOMENT,
    DESC_HU,
    DESC_FD
};

struct BaseFormes {
    Ptr<TrainData> baseDonnees;
    Ptr<TrainData> baseDonneesANN;
    vector<vector<Point>> formes;
    int nbFormeParNiveauBruit = 40;
    int nbNiveauBruit = 8;
    vector<float> minF;
    vector<float> maxF;
};

struct OutilsDessin {
    Mat feuille;
    int tailleCrayon;
    int couleurCrayon;
    bool effaceFeuille=true;
};

void DonneesModele(BaseFormes &);
vector<Point> NoisyPolygon(vector<Point> pRef, double n);
Ptr<ANN_MLP> InitModeleANN(BaseFormes &md);
Ptr<SVM> InitModeleSVM(BaseFormes &md);
Ptr<EM> InitModeleEM(BaseFormes &md);
Ptr<KNearest> InitModeleKNearest(BaseFormes &md);
Ptr<NormalBayesClassifier> InitModeleNormalBayesClassifier(BaseFormes &md);
Ptr<LogisticRegression> InitModeleLogisticRegression(BaseFormes &md);
Ptr<RTrees> InitModeleRTrees(BaseFormes &md);

Mat DescripteurContour(vector<Point> contour, int typeDescripteur= DESC_HU);
void GestionCrayon(int evt, int x, int y, int type, void *extra);
void NormaliseContour(vector<Point> &c, int tailleBoite);

int main()
{
    vector<Ptr<StatModel>> modeleDispo;
    BaseFormes md;

    Ptr<ANN_MLP> modeleANN = InitModeleANN(md);
    if (modeleANN->isTrained())
        modeleDispo.push_back(modeleANN);
    Ptr<EM> modeleEM = InitModeleEM(md);
    if (modeleEM->isTrained())
        modeleDispo.push_back(modeleEM);
    Ptr<KNearest> modeleKNN = InitModeleKNearest(md);
    if (modeleKNN->isTrained())
        modeleDispo.push_back(modeleKNN);
    Ptr<LogisticRegression> modeleLR = InitModeleLogisticRegression(md);
    if (modeleLR->isTrained())
        modeleDispo.push_back(modeleLR);
    Ptr<NormalBayesClassifier> modeleNB = InitModeleNormalBayesClassifier(md);
    if (modeleNB->isTrained())
        modeleDispo.push_back(modeleNB);
    Ptr<SVM> modeleSVM = InitModeleSVM(md);
    if (modeleSVM->isTrained())
        modeleDispo.push_back(modeleSVM);
    Ptr<RTrees> modeleRTrees = InitModeleRTrees(md);
    if (modeleRTrees->isTrained())
        modeleDispo.push_back(modeleRTrees);
    int code = 0;
    String nomFenetre = "Feuille";
    namedWindow(nomFenetre);
    OutilsDessin abcdere;

    abcdere.feuille = Mat::zeros(700, 700, CV_8UC1);
    abcdere.couleurCrayon = 255;
    abcdere.tailleCrayon = 4;
    setMouseCallback(nomFenetre, GestionCrayon, &abcdere);
    do
    {
        imshow("Feuille", abcdere.feuille);
        code = waitKey(10);
        if (code >= '0' && code<'0' + md.formes.size())
        {
            abcdere.feuille.setTo(0);
            if (code - 48<md.formes.size())
                drawContours(abcdere.feuille, md.formes, code - 48, Scalar(abcdere.couleurCrayon), abcdere.tailleCrayon);
        }
        switch(code){
        case 'e':
            abcdere.feuille.setTo(0);
            destroyWindow("formes retenues");
            destroyWindow("Classement");
            break;
        case '+' :
            abcdere.tailleCrayon++;
            if (abcdere.tailleCrayon>10)
                abcdere.tailleCrayon=10;
            break;
        case '-':
            abcdere.tailleCrayon--;
            if (abcdere.tailleCrayon<1)
                abcdere.tailleCrayon=1;
            break;
        case 'c':
            vector<vector<Point>> contours;
            vector<Vec4i> arbreContour;
            findContours(abcdere.feuille, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
            Mat formeRetenue(abcdere.feuille.size(), CV_8UC1,Scalar(0));

            waitKey(10);
            if (contours.size()==1)
            {
                Mat result;
                Mat desc = DescripteurContour(contours[0]);
                desc = desc -Mat(md.minF).t();
                desc =desc.mul(1.0/(Mat(md.maxF).t()-Mat(md.minF).t()));
                cout << "? = " << desc << "\n";
                double minX, maxX;
                Point pos;
                Mat frame(600, 600, CV_8UC1, Scalar(0));
                NormaliseContour(contours[0],100);
                contours.push_back(contours[0]);
                drawContours(formeRetenue, contours, 1, Scalar(255), -1);
                imshow("Formes retenue", formeRetenue);
                for (int i = 0; i < modeleDispo.size(); i++)
                {
                    float classe = modeleDispo[i]->predict(desc, result);
                    if (result.rows*result.cols!=1)
                        minMaxLoc(result, &minX, &maxX, NULL, &pos);
                    else
                    {
                        if (result.type()==CV_32S)
                            pos.x = static_cast<int>(result.at<int>(0, 0));
                        else
                            pos.x = static_cast<int>(result.at<float>(0, 0));
                    }
                    cout << modeleDispo[i]->getDefaultName() << " \tclasse = "<<classe<<" pos= " << pos << "\t"<<result<<"\n";
                    Mat frametmp(700, 700, CV_8UC1, Scalar(0));
                    drawContours(frametmp, md.formes, pos.x, Scalar(255), -1);
                    putText(frametmp, modeleDispo[i]->getDefaultName(),Point(40,40), FONT_HERSHEY_SIMPLEX,2,128,3);
                    Rect rDst(Point(i%3*200,i/3*200),Size(200,200));
                    resize(frametmp,frametmp,Size(200,200));
                    frametmp.copyTo(frame(rDst));
                }
                abcdere.effaceFeuille=true;
                imshow("Classement", frame);
                waitKey(10);
            }
            else
            {
                Mat frametmp(700, 700, CV_8UC1, Scalar(0));
                putText(frametmp, "Erreur plusieurs contours", Point(40, 40), FONT_HERSHEY_SIMPLEX, 2, 128, 3);
                imshow("Cela ressemble ", frametmp);
                waitKey(10);
                abcdere.effaceFeuille = true;
            }
            break;
        }

    } 
    while (code != 27);
return 0;
}

vector<Point> NoisyPolygon(vector<Point> pRef, double n)
{
    RNG rng;
    vector<Point> c;
    vector<Point> p = pRef;
    vector<vector<Point>> contour;
    for (int i = 0; i<p.size(); i++)
        p[i] += Point(static_cast<int>(n*rng.uniform(-1.0, 1.0)), static_cast<int>(n*rng.uniform(-1.0, 1.0)));
    c.push_back(p[0]);
    int minX = p[0].x, maxX = p[0].x, minY = p[0].y, maxY = p[0].y;
    for (int i = 0; i <p.size(); i++)
    {
        int next = i + 1;
        if (next == p.size())
            next = 0;
        Point2d u = p[next] - p[i];
        int d = static_cast<int>(norm(u));
        double a = atan2(u.y, u.x);
        int step = 1;
        if (n != 0)
            step = static_cast<int>(d / n);
        for (int j = 1; j<d; j += max(step, 1))
        {
            Point pNew;
            do
            {
                Point2d pAct = (u*j) / static_cast<double>(d);
                double r = n*rng.uniform((double)0, (double)1);
                double theta = a + rng.uniform(0., 2 * CV_PI);
                pNew = Point(static_cast<int>(r*cos(theta) + pAct.x + p[i].x), static_cast<int>(r*sin(theta) + pAct.y + p[i].y));
            } while (pNew.x<0 || pNew.y<0);
            if (pNew.x<minX)
                minX = pNew.x;
            if (pNew.x>maxX)
                maxX = pNew.x;
            if (pNew.y<minY)
                minY = pNew.y;
            if (pNew.y>maxY)
                maxY = pNew.y;
            c.push_back(pNew);
        }
    }
    contour.push_back(c);
    Mat frame(maxY + 2, maxX + 2, CV_8UC1, Scalar(0));
    drawContours(frame, contour, 0, Scalar(255), -1);
    findContours(frame, contour, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    return contour[0];
}


void DonneesModele(BaseFormes &md)
{
    FileStorage forme("formes.yml", FileStorage::READ);
    if (!forme.isOpened() || forme["PolygoneRef"].empty())
    {
        forme.release();
        vector<Point>  vertex;
 
        vertex.push_back(Point(300, 300));
        vertex.push_back(Point(500, 300));
        vertex.push_back(Point(500, 500));
        md.formes.push_back(vertex);
        vertex.push_back(Point(300, 500));
        md.formes.push_back(vertex);
        FileStorage f("formes.yml", FileStorage::WRITE);
        f << "PolygoneRef" << md.formes;
    }
    else 
    {
        forme["PolygoneRef"] >> md.formes;
        forme.release();
    }
    if (md.nbFormeParNiveauBruit ==0)
        return;
    int nbLigne= static_cast<int>(md.formes.size())*md.nbFormeParNiveauBruit*md.nbNiveauBruit;
    Mat feature;
    Mat responsesANN(nbLigne, static_cast<int>(md.formes.size()), CV_32FC1, Scalar::all(0));
    Mat responses(nbLigne, 1, CV_32FC1, Scalar(0));
    for (int indForme = 0; indForme < md.formes.size(); indForme++)
    {
        int offsetRow = indForme*md.nbFormeParNiveauBruit*md.nbNiveauBruit;
        double noiseLevel = 1;
        for (int i = 0; i < md.nbNiveauBruit; i++)
        {
            for (int j = 0; j < md.nbFormeParNiveauBruit; j++)
            {
                vector<Point> c = NoisyPolygon(md.formes[indForme], noiseLevel);
                Mat desc = DescripteurContour(c);
                feature.push_back(desc);
                responsesANN.at<float>(i*md.nbFormeParNiveauBruit + j + offsetRow, indForme) = 1;
                responses.at<float>(i*md.nbFormeParNiveauBruit + j + offsetRow, 0) = static_cast<float>(indForme);
                if (j % 10 == 0)
                {
                    Mat frame(700, 700, CV_8UC1, Scalar(0));
                    vector<vector<Point>> contour;
                    NormaliseContour(c,100);
                    contour.push_back(c);
                    drawContours(frame, contour, static_cast<int>(contour.size()) - 1, Scalar(255), -1);
                    imshow("Contour Ref.", frame);
                    waitKey(10);
                }

            }
            noiseLevel += 1;
        }
    }
    vector<double> minF(feature.cols), maxF(feature.cols);
    if (md.minF.size()==0)
        for (int i = 0; i < feature.cols; i++)
            {
                minMaxLoc(feature.col(i), minF.data()+i, maxF.data()+i);
                md.minF.push_back(minF[i]);
                md.maxF.push_back(maxF[i]);
            }
    for (int i = 0; i < feature.rows; i++)
    {
        Mat w= feature.row(i) - Mat(md.minF).t();
        feature.row(i) = w.mul(1.0/(Mat(md.maxF).t() -Mat(md.minF).t()));
    }
    Mat typeVariable(feature.cols + 1, 1, CV_8U);
    Mat typeVariableANN(feature.cols + static_cast<int>(md.formes.size()), 1, CV_8U);
    for (int i = 0; i < feature.cols; i++)
    {
        typeVariable.at<uchar>(i, 0) = VAR_NUMERICAL;
        typeVariableANN.at<uchar>(i, 0) = VAR_NUMERICAL;

    }
    for (int i = 0; i<md.formes.size(); i++)
        typeVariableANN.at<uchar>(feature.cols +i, 0) = VAR_NUMERICAL;
    typeVariable.at<uchar>(feature.cols, 0) = VAR_CATEGORICAL;

    md.baseDonnees = TrainData::create(feature, ROW_SAMPLE, responses, noArray(), noArray(), noArray(), typeVariable);
    md.baseDonneesANN = TrainData::create(feature, ROW_SAMPLE, responsesANN, noArray(), noArray(), noArray(), typeVariableANN);
    md.baseDonnees->shuffleTrainTest();
    md.baseDonneesANN->shuffleTrainTest();
    md.baseDonneesANN->setTrainTestSplitRatio(0.8);
    md.baseDonnees->setTrainTestSplitRatio(0.8);
    cout << md.baseDonnees->getNSamples()<< " " << md.baseDonnees->getNTrainSamples() << " " << md.baseDonnees->getNTestSamples() << "\n";
    destroyWindow("Contour Ref.");
}


void GestionCrayon(int evt, int x, int y, int type, void *extra)
{
    OutilsDessin *pgc = (OutilsDessin*)extra;
    if (type == EVENT_FLAG_LBUTTON)
    {
        if (pgc->effaceFeuille)
        {
            pgc->feuille.setTo(0);
            pgc->effaceFeuille = false;
        }
        if (x < pgc->feuille.cols && y < pgc->feuille.cols)
            circle(pgc->feuille, Point(x, y), pgc->tailleCrayon, Scalar(pgc->couleurCrayon), -1);
    }
}

void NormaliseContour(vector<Point> &c,int tailleBoite)
{
    Rect r= boundingRect(c);
    double k= 1.0/max(r.br().x, r.br().y)*(tailleBoite-20);
    for (int i = 0; i < c.size(); i++)
    {
        c[i]=(c[i]-r.tl()+Point(10,10))*k;
    }
}


Mat DescripteurContour(vector<Point> contour, int typeDescripteur)
{
    Mat descripteur;
    Moments m = moments(contour);
    double p;
    switch (typeDescripteur) {
    case DESC_HU:
    {
        descripteur = Mat::zeros(1, 7, CV_32FC1);
        double hu[7];
        HuMoments(m, hu);
        for (int k = 0; k<7; k++)
            descripteur.at<float>(0, k) = static_cast<float>(hu[k]);
       // descripteur.at<float>(0, 7)=m.m00/p/p;
        break;
    }
    case DESC_MOMENT:
        p=arcLength(contour,true);
        descripteur = Mat::zeros(1, 8, CV_32FC1);
        descripteur.at<float>(0, 0) = static_cast<float>(m.nu20);
        descripteur.at<float>(0, 1) = static_cast<float>(m.nu11);
        descripteur.at<float>(0, 2) = static_cast<float>(m.nu02);
        descripteur.at<float>(0, 3) = static_cast<float>(m.nu30);
        descripteur.at<float>(0, 4) = static_cast<float>(m.nu21);
        descripteur.at<float>(0, 5) = static_cast<float>(m.nu12);
        descripteur.at<float>(0, 6) = static_cast<float>(m.nu03);
        descripteur.at<float>(0, 7)= static_cast<float>(m.m00/p/p);
        break;
    case DESC_FD:
#ifdef __OPENCV_FOURIERDESCRIPTORS_HPP__
        ximgproc::fourierDescriptor(contour,descripteur,256,16);
        descripteur.convertTo(descripteur,CV_32F);
        descripteur=descripteur.t();
        descripteur = descripteur.reshape(1);
#else
        cout<<" Descripteur de fourier non défini\n";
        exit(0);
#endif
        break;

    }
    return descripteur;
}

Ptr<ANN_MLP> InitModeleANN(BaseFormes &md)
{
    Ptr<ANN_MLP> machine = Algorithm::load<ANN_MLP>("opencv_ml_ann_mlp.yml");
    if (!machine)
    {
        if (md.formes.size() == 0)
            DonneesModele(md);
        machine = ANN_MLP::create();
        Mat_<int> layerSizes(1, 3);
        layerSizes(0, 0) = md.baseDonneesANN->getNVars();
        layerSizes(0, 1) = md.baseDonneesANN->getNVars();
        layerSizes(0, 2) = static_cast<int>(md.formes.size());
        machine->setLayerSizes(layerSizes);
        machine->setActivationFunction(ANN_MLP::SIGMOID_SYM);
        machine->setTrainMethod(ANN_MLP::RPROP);
        machine->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100000, 0.0001));
        machine->train(md.baseDonneesANN);
        if (machine->isTrained())
        {
            Mat reponse;
            cout << "Modele " << machine->getDefaultName() << "\n";
            cout << "Erreur sur les donnees tests : " << machine->calcError(md.baseDonneesANN, true, noArray()) << "\n";
            cout << "Erreur sur les donnees d'entrainement : " << machine->calcError(md.baseDonneesANN, false, noArray()) << "\n";
        }
        else
        {
            cout << "ERREUR lors de l'initialisation du modèle\n ";
            return machine;

        }
        machine->save(machine->getDefaultName() + ".yml");
        FileStorage fs(machine->getDefaultName() + ".yml", FileStorage::APPEND);
        fs << "minF" << md.minF;
        fs << "maxF" << md.maxF;
    }
    else if (md.minF.size()==0)
    {
        FileStorage fs(machine->getDefaultName() + ".yml", FileStorage::READ);
        fs["minF"] >> md.minF;
        fs["maxF"] >> md.maxF;
    }
    return machine;
}

Ptr<SVM> InitModeleSVM(BaseFormes & md)
{
    Ptr<SVM> machine = Algorithm::load<SVM>("opencv_ml_svm.yml");

    if (!machine)
    {
        if (md.formes.size() == 0)
            DonneesModele(md);
        machine = SVM::create();
        machine->setKernel(SVM::RBF);
        machine->setType(SVM::C_SVC);
        machine->setGamma(1); 
        machine->setC(1); 
        machine->trainAuto(md.baseDonnees);
        if (machine->isTrained())
        {
            Mat reponse;
            cout << "Modele " << machine->getDefaultName() << "\n";
            cout << "Erreur sur les donnees tests : " << machine->calcError(md.baseDonnees, true, noArray()) << "\n";
            cout << "Erreur sur les donnees d'entrainement : " << machine->calcError(md.baseDonnees, false, noArray()) << "\n";
        }
        else
        {
            cout << "ERREUR lors de l'initialisation du modèle\n ";
            return machine;
        }
        machine->save(machine->getDefaultName() + ".yml");
        FileStorage fs(machine->getDefaultName() + ".yml", FileStorage::APPEND);
        fs << "minF" << md.minF;
        fs << "maxF" << md.maxF;
    }
    else if (md.minF.size() == 0)
    {
        FileStorage fs(machine->getDefaultName() + ".yml", FileStorage::READ);
        fs["minF"] >> md.minF;
        fs["maxF"] >> md.maxF;
    }
    return machine;
}

Ptr<NormalBayesClassifier> InitModeleNormalBayesClassifier(BaseFormes & md)
{
    Ptr<NormalBayesClassifier> machine = Algorithm::load<NormalBayesClassifier>("opencv_ml_nbayes.yml");

    if (!machine)
    {
        if (md.formes.size() == 0)
            DonneesModele(md);
        machine = NormalBayesClassifier::create();
        machine->train(md.baseDonnees);
        if (machine->isTrained())
        {
            Mat reponse;
            cout << "Modele " << machine->getDefaultName() << "\n";
            cout << "Erreur sur les donnees tests : " << machine->calcError(md.baseDonnees, true, noArray()) << "\n";
            cout << "Erreur sur les donnees d'entrainement : " << machine->calcError(md.baseDonnees, false, noArray()) << "\n";
            machine->save(machine->getDefaultName() + ".yml");
            FileStorage fs(machine->getDefaultName() + ".yml", FileStorage::APPEND);
            fs << "minF" << md.minF;
            fs << "maxF" << md.maxF;
        }
        else
        {
            cout << "ERREUR lors de l'initialisation du modèle\n ";
            return machine;
        }
    }
    else if (md.minF.size() == 0)
    {
        FileStorage fs(machine->getDefaultName() + ".yml", FileStorage::READ);
        fs["minF"] >> md.minF;
        fs["maxF"] >> md.maxF;
    }
return machine;
}

Ptr<KNearest> InitModeleKNearest(BaseFormes & md)
{
    Ptr<KNearest> machine = Algorithm::load<KNearest>("opencv_ml_knn.yml");

    if (!machine)
    {
        if (md.formes.size() == 0)
            DonneesModele(md);
        machine = KNearest::create();
        machine->setIsClassifier(true);
        machine->setAlgorithmType(KNearest::BRUTE_FORCE);
        machine->train(md.baseDonnees);
        if (machine->isTrained())
        {
            Mat reponse;
            cout << "Modele " << machine->getDefaultName() << "\n";
            cout << "Erreur sur les donnees tests : " << machine->calcError(md.baseDonnees, true, noArray()) << "\n";
            cout << "Erreur sur les donnees d'entrainement : " << machine->calcError(md.baseDonnees, false, noArray()) << "\n";
            machine->save(machine->getDefaultName() + ".yml");
            FileStorage fs(machine->getDefaultName() + ".yml", FileStorage::APPEND);
            fs << "minF" << md.minF;
            fs << "maxF" << md.maxF;
        }
        else
        {
            cout << "ERREUR lors de l'initialisation du modèle\n ";
            return machine;

        }
    }
    else if (md.minF.size() == 0)
    {
        FileStorage fs(machine->getDefaultName() + ".yml", FileStorage::READ);
        fs["minF"] >> md.minF;
        fs["maxF"] >> md.maxF;
    }
    return machine;
}


Ptr<LogisticRegression> InitModeleLogisticRegression(BaseFormes & md)
{
    Ptr<LogisticRegression> machine = Algorithm::load<LogisticRegression>("opencv_ml_lr1.yml");

    if (!machine)
    {
        if (md.formes.size() == 0)
            DonneesModele(md);
        machine = LogisticRegression::create();
        machine->setLearningRate(0.1);
        machine->setIterations(1000);
        machine->setRegularization(LogisticRegression::REG_L2);
        machine->setTrainMethod(LogisticRegression::BATCH);
        machine->setMiniBatchSize(1);
        machine->train(md.baseDonnees);
        if (machine->isTrained())
        {
            Mat reponse;
            cout << "Modele " << machine->getDefaultName() << "\n";
            cout << "Erreur sur les donnees tests : " << machine->calcError(md.baseDonnees, true, noArray()) << "\n";
            cout << "Erreur sur les donnees d'entrainement : " << machine->calcError(md.baseDonnees, false, noArray()) << "\n";
            machine->save(machine->getDefaultName() + ".yml");
            FileStorage fs(machine->getDefaultName() + ".yml", FileStorage::APPEND);
            fs << "minF" << md.minF;
            fs << "maxF" << md.maxF;
        }
        else
        {
            cout << "ERREUR lors de l'initialisation du modèle\n ";
            return machine;
        }
    }
    else if (md.minF.size() == 0)
    {
        FileStorage fs(machine->getDefaultName() + ".yml", FileStorage::READ);
        fs["minF"] >> md.minF;
        fs["maxF"] >> md.maxF;
    }
    return machine;
}


Ptr<EM> InitModeleEM(BaseFormes & md)
{
    Ptr<EM> machine = Algorithm::load<EM>("opencv_ml_em.yml");

    if (!machine)
    {
        if (md.formes.size() == 0)
            DonneesModele(md);
        machine = EM::create();
        machine->setClustersNumber(static_cast<int>(md.formes.size()));
        machine->setCovarianceMatrixType(EM::COV_MAT_DIAGONAL);
        machine->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 1000, 0.1));
        machine->train(md.baseDonnees);
        if (machine->isTrained())
        {
            Mat reponse;
            cout << "Modele " << machine->getDefaultName() << "\n";
            cout << "Erreur sur les donnees tests : " << machine->calcError(md.baseDonnees, true, noArray()) << "\n";
            cout << "Erreur sur les donnees d'entrainement : " << machine->calcError(md.baseDonnees, false, noArray()) << "\n";
            machine->save(machine->getDefaultName() + ".yml");
            FileStorage fs(machine->getDefaultName() + ".yml", FileStorage::APPEND);
            fs << "minF" << md.minF;
            fs << "maxF" << md.maxF;
        }
        else
        {
            cout << "ERREUR lors de l'initialisation du modèle\n ";
            return machine;
        }
    }
    else if (md.minF.size() == 0)
    {
        FileStorage fs(machine->getDefaultName() + ".yml", FileStorage::READ);
        fs["minF"] >> md.minF;
        fs["maxF"] >> md.maxF;
    }
    return machine;
}

Ptr<RTrees> InitModeleRTrees(BaseFormes &md)
{

    Ptr<RTrees> machine = Algorithm::load<RTrees>("opencv_ml_rtrees.yml");
    if (!machine)
    {
        if (md.formes.size() == 0)
            DonneesModele(md);
        if (md.formes.size() == 0)
        {
            cout << "ERREUR aucune forme définie!\n";
            exit(0);
        }
        machine = RTrees::create();
        machine->train(md.baseDonnees);
        if (machine->isTrained())
        {
            Mat reponse;
            cout << "Modele " << machine->getDefaultName() << "\n";
            cout << "Erreur sur les donnees tests : " << machine->calcError(md.baseDonnees, true, noArray()) << "\n";
            cout << "Erreur sur les donnees d'entrainement : " << machine->calcError(md.baseDonnees, false, noArray()) << "\n";
            machine->save(machine->getDefaultName() + ".yml");
            FileStorage fs(machine->getDefaultName() + ".yml", FileStorage::APPEND);
            fs << "minF" << md.minF;
            fs << "maxF" << md.maxF;
        }
        else
        {
            cout << "ERREUR lors de l'initialisation du modèle\n ";
            return machine;

        }
    }
    else if (md.minF.size() == 0)
    {
        FileStorage fs(machine->getDefaultName() + ".yml", FileStorage::READ);
        fs["minF"] >> md.minF;
        fs["maxF"] >> md.maxF;
    }
    return machine;
}
