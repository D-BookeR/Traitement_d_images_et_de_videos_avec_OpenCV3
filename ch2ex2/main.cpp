#include <opencv2/opencv.hpp> 
#include <iostream>

using namespace cv;
using namespace std;

#define NBCAMERA 10

int indCamActive=-1;
int ksize=7;

const String keys =
"{Aide h usage ? help  |     | Afficher ce message   }"
"{@arg1                |     | nom du fichier de configuration}"             
"{c0                   |     | widthXheight   }"
"{c1                   |     | widthXheight   }"
"{c2                   |     | widthXheight   }"
"{c3                   |     | widthXheight   }"
"{c4                   |     | widthXheight   }"
"{c5                   |     | widthXheight   }"
"{c6                   |     | widthXheight   }"
"{c7                   |     | widthXheight   }"
"{c8                   |     | widthXheight   }"
"{c9                   |     | widthXheight   }"
"{f                    |config.xml| sauvegarde de la configuration en xml oy yml     }"
"{d                    | | Affichage des webcam sur une seule image   }"
"{z                    |1:1 | zoom pour toutes les images   }"
;



static map<int, VideoCapture> OuvrirLesFlux(CommandLineParser);
static map<int, VideoCapture> OuvrirLesFlux(String);
static int  SauverFichierConfiguration(String , map<int, VideoCapture> );
static bool FixeTailleFlux(VideoCapture , Size );
static double OptionZoom(CommandLineParser parser);
static void onMouse(int event, int x, int y, int flags, void *f);


int main (int argc,char **argv)
{
    map<int ,VideoCapture> v;
    CommandLineParser parser(argc, argv, keys);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    String nomFic = parser.get<String>(0);
    if (nomFic.length() > 0)
    {
        v = OuvrirLesFlux(nomFic);
    }
    else
        v = OuvrirLesFlux(parser);
    if (v.size() == 0)
    {
        cout << "Aucun périphérique vidéo ouvert vérifier votre système ou la ligne de commande\n";
        return 0;
    }
    if (parser.has("f"))
    {
        String nomFic=parser.get<String>("f");
        if (SauverFichierConfiguration(nomFic,v))
            cout<<"Impossible de sauvegarder le fichier de configuration\n";
    }
    bool affichageUnique=false;
    vector<Rect> rCam;
    double zoom=1;
    if (parser.has("z"))
    {
        zoom=OptionZoom(parser);
    }
    Mat frameUnique;
    if (parser.has("d"))
    {
        affichageUnique=true;
        Point p(0,0);
        map<int,VideoCapture>::iterator ite=v.begin();
        for (size_t i = 0; i<v.size(); i++,ite++)
        {
            Size s(static_cast<int>(round(ite->second.get(CAP_PROP_FRAME_WIDTH)*zoom)), static_cast<int>(round(ite->second.get(CAP_PROP_FRAME_HEIGHT)*zoom)));
            rCam.push_back(Rect(Point(0,p.y),s));
            p.y += s.height;
            p.x=max(p.x,s.width);

        }
        frameUnique =Mat::zeros(p.y, p.x, CV_8UC3);
        namedWindow("All WebCam");
        setMouseCallback("All WebCam", onMouse,&rCam);
    }

    vector<Mat> frame(v.size());
    char c=0;
    int nbCapture=0;
    int64 tpsIni=getTickCount();
    bool calculLaplacien=false;
    for (;c!=27;)
    {
        if (affichageUnique)
        {
            map<int, VideoCapture>::iterator ite = v.begin();
            for (size_t i = 0; i<v.size() && c != 27; i++,ite++)
            {
                ite->second >> frame[i];
                if (calculLaplacien)
                    Laplacian(frame[i],frame[i],-1, 2 * ksize + 1);
                if (zoom != 1)
                {
                    Mat frameZoom;
                    resize(frame[i], frameZoom, Size(), zoom, zoom);
                    frameZoom.copyTo(frameUnique(rCam[i]));
                }
                else
                    frame[i].copyTo(frameUnique(rCam[i]));
            }
            imshow("All WebCam", frameUnique);
            if (indCamActive>=0)
                imshow("Cam Select", frame[indCamActive]);
        }
        else
        {
            map<int, VideoCapture>::iterator ite = v.begin();
            for (size_t i = 0; i<v.size() && c!=27;i++,ite++)
            {
                ite->second >> frame[i];
                if (calculLaplacien)
                    Laplacian(frame[i], frame[i], -1, 2*ksize+1);
                if (zoom != 1)
                {
                    Mat frameZoom;
                    resize(frame[i], frameZoom,Size(),zoom,zoom);
                    imshow(format("Main WebCam %d", i), frameZoom);
                }
                else
                    imshow(format("Main WebCam %d", i), frame[i]);   
            }
        }
        c=waitKey(1);
        switch (c) 
        {
        case 'l' :
            calculLaplacien = false;
            break;
        case 'L' :
            calculLaplacien = true;
            break;
        case '+' :
            ksize++;
            break;
        case '-' :
            if (ksize>1)
                ksize--;
            break;
        }
        nbCapture +=static_cast<int>(v.size());
        if (nbCapture > 500)
        {
            int64 tpsFin=getTickCount();
            cout<< "image par seconde :"<< nbCapture/((tpsFin-tpsIni)/getTickFrequency())<<"\n";
            cout << "ksize :" << ksize << "\n";
            nbCapture=0;
            tpsIni = getTickCount();
        }
    }
return 0;
}

static  map<int, VideoCapture> OuvrirLesFlux(CommandLineParser parser)
{
    map<int, VideoCapture> v;
    for (int i = 0; i < NBCAMERA; i++)
    {
        String option = format("c%d", i);
        if (parser.has(option))
        {
            String s = parser.get<String>(option);
            VideoCapture vid(i);
            if (vid.isOpened())
            {
                size_t pos = s.find_first_of("x");
                if (pos == String::npos)
                    pos = s.find_first_of("X");
                if (pos != String::npos)
                {
                    int width = stoi(s.substr(0, pos));
                    int height = stoi(s.substr(pos + 1, s.length() - pos + 1));
                    bool status = FixeTailleFlux(vid, Size(width, height));
                    if (!status)
                    {
                        cout << "****************** ATTENTION ******************\n";
                        cout << "Format " << s << " pour la caméra " << i << " n'est pas disponible.\n";
                        cout << "Format le plus proche : " << vid.get(CAP_PROP_FRAME_WIDTH) << "X" << vid.get(CAP_PROP_FRAME_HEIGHT) << "\n";
                        cout << "***********************************************\n";
						vid.release();
					}
                    else
                        v.insert(make_pair(i, vid));
                }
                else
                {
                    cout << "****************** ATTENTION ******************\n";
                    cout << "Erreur de syntaxe option  " << option << " : " << s << "\n";
                    vid.release();
                }
            }
            else
                cout << "Caméra " << i << " indisponible.\n";
        }
    }
    return v;
}

bool FixeTailleFlux(VideoCapture vid, Size s)
{
    bool status = vid.set(CAP_PROP_FRAME_WIDTH, s.width);
    status = status && vid.set(CAP_PROP_FRAME_HEIGHT, s.height);
    if (!status || s.height != vid.get(CAP_PROP_FRAME_HEIGHT) || s.width != vid.get(CAP_PROP_FRAME_WIDTH))
    {
        status = false;

    }
    return status;

}

static map<int, VideoCapture> OuvrirLesFlux(String nomFic)
{
    map<int, VideoCapture> v;
    FileStorage fs(nomFic, FileStorage::READ);
    if (fs.isOpened())
    {
        FileNode n = fs["cam"];
        if (!n.empty())
        {
            FileNodeIterator it = n.begin();
            while (it != n.end())
            {
                int i;
                Size s;
                (*it) >> i;
                it++;
                (*it) >> s;
                it++;
                if (v.find(i) == v.end())
                {
                    VideoCapture vid(i);
                    if (vid.isOpened() && FixeTailleFlux(vid,s))
                        v.insert(make_pair(i, vid));
                    else
                        vid.release();
                }
                else
                    cout<< "Le flux est déjà ouvert\n";
            }
        }
    }
    fs.release();
    return v;
}

int  SauverFichierConfiguration(String nomFic, map<int, VideoCapture> v)
{

    FileStorage fs(nomFic, FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "cam" << "[";
        map<int, VideoCapture>::iterator ite = v.begin();

        for (; ite !=v.end(); ite++)
        {
            fs << ite->first << Size(static_cast<int>(ite->second.get(CAP_PROP_FRAME_WIDTH)), static_cast<int>(ite->second.get(CAP_PROP_FRAME_HEIGHT)));
        }
        fs << "]";
        fs.release();
        return 0;
    }
    return 1;
}

double OptionZoom(CommandLineParser parser)
{
    double zoom = 1;
    String facteurZoom = parser.get<String>("z");
    size_t pos = facteurZoom.find_first_of(":");
    if (pos == String::npos || pos == 0)
        zoom = 1;
    else
    {
        int numerateur = stoi(facteurZoom.substr(0, pos));
        int denominateur = 1;
        if (pos<facteurZoom.length() - 1)
            denominateur = stoi(facteurZoom.substr(pos + 1, facteurZoom.length() - pos + 1));
        if (denominateur == 0)
        {
            cout << "Paramètre zoom incorrect. Valeur par défaut utilisée.\n";
        }
        else
            zoom = static_cast<double>(numerateur) / static_cast<double>(denominateur);
    }
    return zoom;
}

static void onMouse(int event, int x, int y, int flags, void *f)
{
    vector<Rect> *rCam = static_cast<vector<Rect> *>(f);

    if (event == CV_EVENT_LBUTTONDOWN)
    {
        Point p(x, y);
        indCamActive = -1;
        for (size_t i = 0; i < rCam->size(); i++)
        {
            if (p.inside((*rCam)[i]))
            {
                indCamActive = i;
            }
        }

    }
}
