myRepo=$(pwd)
CMAKE_CONFIG_GENERATOR="Visual Studio 14 2015 Win64"
echo $myRepo
echo $myIDE
if [  ! -d "$myRepo/opencv"  ]; then
  echo "clonning opencv"
  git clone https://github.com/opencv/opencv.git
  mkdir Build
  mkdir Build/opencv
  mkdir Install/opencv
else
  cd opencv
  git pull
  cd ..  
fi
if [  ! -d "$myRepo/opencv_contrib"  ]; then
  echo "clonning opencv_contrib"
  git clone https://github.com/opencv/opencv_contrib.git
  mkdir Build
  mkdir Build/opencv_contrib
else
  cd opencv_contrib
  git pull
  cd ..  
fi
RepoSource=opencv
cd Build/$RepoSource
optCMAKE=" "

cmake -G"$CMAKE_CONFIG_GENERATOR" -DBUILD_EXAMPLES:BOOL=OFF -DBUILD_DOCS=OFF -DBUILD_PERF_TESTS:BOOL=OFF -DBUILD_TESTS:BOOL=OFF -DWITH_CUDA:BOOL=OFF -DCMAKE_INSTALL_PREFIX=../../install/"$RepoSource" -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ../../"$RepoSource" 

cd ..
echo "************************* $Source_DIR -->devenv debug"
cmake --build $RepoSource  --config debug 
echo "************************* $Source_DIR -->devenv release"
cmake --build $RepoSource  --config release 
cmake --build $RepoSource  --target install --config release 
cmake --build $RepoSource  --target install --config debug 
cd ..
