#include <opencv2/opencv.hpp> 
#include <opencv2/face.hpp>
#include <opencv2/plot.hpp>

using namespace std;
using namespace cv;


// http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
const String keys =
"{Aide h usage ? help  |     | Afficher ce message   }"
"{d                    |     | nom du dossier à explorer }"
"{b                    | 0   | 0 pour base video 1  pour base att }"
"{l                    | 0   | 1  pour algorithme lbph }"
"{f                    | 0   | 1  pour algorithme Fischerface }"
"{e                    | 0   | 1  pour algorithme EigenFace }";

struct BaseImage
{
    vector<Mat> visages;
    vector<int> labels;
    Mat labelsTrain;
    map<int, int> posLabel;
};

int LectureBaseFormatATT(String masque, BaseImage &b); //-d=f:/lib/opencv_extra/att_faces/*.pgm -b=1
int LectureBaseFormatVideo(String masque, BaseImage &b); // -d=F:/LivreOpencv/OpenCV/BuildLivre/faciale/buildVideo/*.jpg -b=0
Mat AfficheBase(BaseImage b, int index);
Mat AfficheImageBase(BaseImage b, int label,int indexDsLabel);
Mat VisualiseLbph(Ptr<face::FaceRecognizer> m, int index);
int LabelSuivant(BaseImage &b, int label);
int LabelPrecedent(BaseImage &b, int label);

int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, keys);
    String nomDossierData;
    String masque;
    BaseImage b;

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    if (parser.has("d"))
    {
        masque = parser.get<String>("d");
    }
    else
    {
        cout<<" Nom du dossier absent\n";
        exit(-1);
    }
    if (parser.get<int>("b")==1)
        LectureBaseFormatATT(masque,b);
    else
        LectureBaseFormatVideo(masque, b);


    vector <String> nomFichiers;
    String window_name="rr";
    vector<Ptr<face::FaceRecognizer>> modelVisage;
    Ptr<face::BasicFaceRecognizer> modelFisher = Algorithm::load<face::FisherFaceRecognizer>("fisherface.xml");
    Ptr<face::BasicFaceRecognizer> modelEigen = Algorithm::load<face::EigenFaceRecognizer>("eigenface.xml");
    Ptr<face::LBPHFaceRecognizer> modelLBPHF = Algorithm::load<face::LBPHFaceRecognizer>("myVideoFace.yml");
    if (modelLBPHF != NULL)
        modelVisage.push_back(modelLBPHF);
    if (modelFisher != NULL)
        modelVisage.push_back(modelFisher);
    if (modelEigen != NULL)
        modelVisage.push_back(modelEigen);
    int code=0;
    int indexImageDsLabel = 0; // Index de l'image dans la liste des images de même label
    int indexSelect=0; // index de la première image du label
    int labelSelect = b.labels[indexSelect];
    int indexBase = 0; // Image représentant label pour la base de données
    int labelBase = b.labels[indexBase];
    do
    {
        Mat img = AfficheBase(b, labelBase);
        imshow("base", img);
        img = AfficheImageBase(b, labelSelect, indexImageDsLabel);
        String nomFenetre= format("Label %d", labelSelect);
        if (!img.empty())
            imshow(nomFenetre, img);
        code = waitKey();
        switch(code){
        case 'U':
            if (indexImageDsLabel<b.visages.size()-1)
                indexImageDsLabel++;
            break;
        case 'D':
            if (indexImageDsLabel>0)
                indexImageDsLabel--;
            break;
        case 'u':
            indexSelect = LabelSuivant(b, labelSelect);
            labelSelect = b.labels[indexSelect];
            indexImageDsLabel = 0;
            destroyWindow(nomFenetre);
            break;
        case 'd':
            indexSelect = LabelPrecedent(b, labelSelect);
            labelSelect = b.labels[indexSelect];
            indexImageDsLabel = 0;
            destroyWindow(nomFenetre);
            break;
        case 'r':
            b.visages.erase(b.visages.begin()+ b.posLabel.find(labelSelect)->second + indexImageDsLabel);
            b.labels.erase(b.labels.begin()+b.posLabel.find(labelSelect)->second + indexImageDsLabel);
            break;
        case '+':
            indexBase = LabelSuivant(b, labelBase);
            labelBase = b.labels[indexBase];
            break;
        case '-':
            indexBase = LabelPrecedent(b, labelBase);
            labelBase = b.labels[indexBase];
            break;
        case 'm':
        {
            int pos = min(b.posLabel.find(labelSelect)->second + indexImageDsLabel, b.posLabel.find(labelBase)->second);
            b.posLabel.find(labelBase)->second=pos;
            b.labels[indexSelect+ indexImageDsLabel] = labelBase;
            destroyWindow(nomFenetre);
        }
        break;
        case 'M':
        {
            int pos = min(b.posLabel.find(labelSelect)->second + indexImageDsLabel, b.posLabel.find(labelBase)->second);
            for (int i= b.posLabel.find(labelSelect)->second;i<b.labels.size();i++)
            {
                if (b.labels[i]== labelSelect)
                    b.labels[i] = labelBase;
            }
            b.posLabel.find(labelBase)->second = pos;
            indexSelect = LabelPrecedent(b, labelSelect);
            labelSelect = b.labels[indexSelect];
            destroyWindow(nomFenetre);
        }
        break;
        case 'f':
            if (modelFisher.empty())
            {
                modelFisher = face::FisherFaceRecognizer::create();
                modelFisher->getDefaultName();
                try
                {
                    modelFisher->train(b.visages, b.labels);
                    modelFisher->save(modelFisher->getDefaultName() + "0.yml");
                    modelVisage.push_back(modelFisher);
                }
                catch (cv::Exception &e)
                {
                    cout << "Cause possible d'erreur : les images ne sont pas de même taille\n";
                    modelFisher.release();
                }
            }
            else
                modelVisage.push_back(modelFisher);
            break;
        case 'e':
            if (modelEigen.empty())
            {
                modelEigen = face::EigenFaceRecognizer::create();
                modelEigen->setNumComponents(80);
                try
                {
                    modelEigen->train(b.visages, b.labels);
                    modelEigen->save(modelEigen->getDefaultName() + "1.yml");
                    modelVisage.push_back(modelEigen);
                }
                catch (cv::Exception &e)
                {
                    cout << "Cause possible d'erreur : les images ne sont pas de même taille\n";
                    modelEigen.release();
                }
            }
            else
                modelVisage.push_back(modelEigen);
            break;
        case 'l':
            modelLBPHF = face::LBPHFaceRecognizer::create();
            try
            {
                modelLBPHF->train(b.visages, b.labels);
                modelLBPHF->save(modelLBPHF->getDefaultName()+"2.yml");
                modelVisage.push_back(modelLBPHF);
            }
            catch (cv::Exception &e)
            {
                modelLBPHF.release();
            }
            break;
        case 's':
            for (int i = 0; i < b.visages.size(); i++)
            {
                if (b.labels[i] != -1)
                {
                    String nom=format("baseVisages/visage_%d_%d.jpg", b.labels[i],i);
                    imwrite(nom,b.visages[i]);
                }
            }
            break;
        }
    }
    while (code!=27);
    cout<<"\n";
    return 0;
}

int LabelSuivant(BaseImage &b, int label)
{
    int i = b.posLabel.find(label)->second;
    while (i < b.labels.size() && (b.labels[i] == label || b.posLabel.find(b.labels[i])->second != i))
    {
        i++;
    }
    if (i== b.labels.size())
        return 0;
    return i;
}

int LabelPrecedent(BaseImage &b, int label)
{
    int i = b.posLabel.find(label)->second;
    while (i >0 && (b.labels[i] == label || b.posLabel.find(b.labels[i])->second != i))
    {
        i--;
    }
    return i;
}

int LectureBaseFormatATT(String masque, BaseImage &b)
{
    vector <String> nomFichiers;
    glob(masque, nomFichiers, true);
    for (int i = 0; i < 320; i++)
    {
        nomFichiers.erase(nomFichiers.begin());
    }
    for (int i = 0; i < nomFichiers.size(); i++)
    {
        std::string::size_type pos;
        string nom(nomFichiers[i]);
        do
        {
            pos = nom.find('\\');
            if (pos != std::string::npos)
                nom[pos] = '/';

        } while (pos != std::string::npos);
        nomFichiers[i] = nom;
    }
    String nomDossier;
    std::string::size_type pos = nomFichiers[0].find_last_of('/');
    if (pos == std::string::npos)
    {
        cout << "Les images doivent être rangées dans un dossier\n";
        return 0;
    }
    int i = 1;
    int nbProfil = 1;
    nomDossier = nomFichiers[0].substr(0, pos + 1);
    b.visages.push_back(imread(nomFichiers[0], IMREAD_ANYCOLOR));
    pos = nomDossier.substr(0, nomDossier.length() - 1).find_last_of('/');
    int indLabel = stoi(nomDossier.substr(pos + 2, nomDossier.length() - pos - 3).c_str());
    b.labels = Mat(1, 1, CV_32SC1, Scalar(indLabel));
    int nbLabel = 0;
    while (i < nomFichiers.size())
    {
        pos = nomFichiers[i].find(nomDossier);
        if (pos == std::string::npos)
        {
            cout << "Fin exploration du dossier : " << nomDossier << "(" << nbProfil << ")\n";
            do
            {
                pos = nomFichiers[i].find_last_of('/');
                i++;
            } while (i<nomFichiers.size() && pos == std::string::npos);
            if (pos != std::string::npos)
            {
                nbProfil = 1;
                nomDossier = nomFichiers[i - 1].substr(0, pos + 1);
                pos = nomDossier.substr(0, nomDossier.length() - 1).find_last_of('/');
                indLabel = stoi(nomDossier.substr(pos + 2, nomDossier.length() - pos - 3).c_str());
                b.labels.push_back(indLabel);
                b.visages.push_back(imread(nomFichiers[i - 1], IMREAD_ANYCOLOR));
                if (b.posLabel.find(indLabel) == b.posLabel.end())
                {
                    b.posLabel.insert(pair<int, int>(indLabel, i-1));
                }
            }
        }
        else
        {
            b.labels.push_back(indLabel);
            b.visages.push_back(imread(nomFichiers[i], IMREAD_ANYCOLOR));
            if (b.posLabel.find(indLabel) == b.posLabel.end())
            {
                b.posLabel.insert(pair<int, int>(indLabel, i ));
            }
            i++;
            nbProfil++;
        }

    }
    return b.visages.size();
}

int LectureBaseFormatVideo(String masque, BaseImage &b)
{

    vector <String> nomFichiers;
    glob(masque, nomFichiers, false);
    int indLabel = 0;
    String classe;
    std::string::size_type pos = nomFichiers[0].find_last_of('_');
    if (pos == std::string::npos)
    {
        cout << "Les images doivent contenir un _ après la classe\n";
        return 0;
    }
    b.labels = Mat(0, 1, CV_32SC1);
    int nbLabel=0;
    for (int i = 0; i < nomFichiers.size(); i++)
    {
        std::string::size_type pos;
        string nom(nomFichiers[i]);
        do
        {
            pos = nom.find('\\');
            if (pos != std::string::npos)
                nom[pos] = '/';

        } while (pos != std::string::npos);
        nomFichiers[i] = nom;
        std::string::size_type pos1 = nomFichiers[i].find_last_of('_');
        std::string::size_type pos2 = nomFichiers[i].find("visage");
        if (pos1 != std::string::npos)
        {
            classe = nomFichiers[i].substr(pos2 + 7, pos1 - pos2 - 6);
            int index = stoi(classe.c_str());
            b.labels.push_back(index);
            b.visages.push_back(imread(nomFichiers[i], IMREAD_ANYCOLOR));
            if (b.posLabel.find(index) == b.posLabel.end())
            {
                b.posLabel.insert(pair<int, int>(index, i)); // position de la première image avec label index
            }
        }
    }
    return b.visages.size();
}

Mat AfficheImageBase(BaseImage b, int label, int indexDsLabel)
{
    int hauteurMax = (750 / b.visages[0].rows*2)*b.visages[0].rows/2;
    int largeurMax = (750 / b.visages[0].cols*2)*b.visages[0].cols/2;
    Mat img(hauteurMax, largeurMax, CV_8UC1, Scalar(0));
    map<int, int>::iterator ite = b.posLabel.find(label);
    if (ite == b.posLabel.end())
        return Mat();
    int N1 = (750 / b.visages[0].rows * 2);
    int N = (750 / b.visages[0].rows * 2)*(750 / b.visages[0].cols * 2);
    int i= b.posLabel.find(label)->second;
    for (int j=0; j<indexDsLabel && i<b.labels.size(); i++)
        if (b.labels[i] == label)
            j++;
    for (int j = 0; j<N && i<b.labels.size(); i++)
        if (b.labels[i] == label)
        {
            Mat x;
            resize(b.visages[i], x, Size(), 0.5, 0.5);
            x.copyTo(img(Rect((j%N1)*x.cols, (j / N1)*x.rows, x.cols, x.rows)));
            j++;
        }
return img;
}

Mat AfficheBase(BaseImage b, int label)
{
    map<int,int>::iterator ite=b.posLabel.find(label);
    int nbL=0,nbC=0;
    if (ite==b.posLabel.end())
        return Mat();
    int largeur=0;
    int hauteur=0;
    int hauteurMax = (750 / b.visages[0].rows)*b.visages[0].rows;
    int largeurMax = (750 / b.visages[0].cols)*b.visages[0].cols;
    Mat img(hauteurMax,largeurMax,CV_8UC1,Scalar(0));
    largeurMax=0;
    hauteurMax=0;
    largeur=0;
    hauteur=0;
    int i = ite->second;
    int labelAct = label-1;
    while (i < b.labels.size() && hauteur < 750)
    {
        if (b.labels[i]!= labelAct && b.posLabel.find(b.labels[i])->second==i)
        {
            int labelAct = b.labels[i];
            map<int, int>::iterator iteb = b.posLabel.find(labelAct);
            b.visages[iteb->second].copyTo(img(Rect(largeur,hauteur, b.visages[iteb->second].cols, b.visages[iteb->second].rows)));
            putText(img,format("%d", labelAct),Point(largeur+10,hauteur+30),FONT_HERSHEY_COMPLEX,1,Scalar(255),1);
            if (labelAct == label)
            {
                rectangle(img, Rect(largeur + 1, hauteur + 1, b.visages[iteb->second].cols - 2, b.visages[iteb->second].rows - 2), Scalar(255), 3);
            }
            if (hauteur + b.visages[iteb->second].rows> hauteurMax)
                hauteurMax = hauteur + b.visages[iteb->second].rows;
            largeur += b.visages[iteb->second].cols;
            if (largeur > 500)
            {
                nbC++;
                hauteur = hauteurMax;
                if (largeur>largeurMax)
                    largeurMax = largeur;
                largeur = 0;
            }
        }
        i++;
    }

    return img;
}
