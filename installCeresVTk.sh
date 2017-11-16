function MAJGitRepo
# $1=nom $2 $3=url
{
if [  ! -d "$myRepo/$1"  ]; then
  echo "clonning ${1}"
  git clone $2
  mkdir Build/$1
else
  echo "update $1"
  cd $1
  git pull
  cd ..  
fi

}
#!/bin/bash
echo "Installation de ceres et vtk"
myRepo=$(pwd)
CMAKE_CONFIG_GENERATOR="Visual Studio 14 2015 Win64"
echo  $CMAKE_CONFIG_GENERATOR 

MAJGitRepo eigen https://github.com/RLovelett/eigen.git

RepoSource=gflags
MAJGitRepo $RepoSource https://github.com/gflags/gflags.git
cd Build/$RepoSource
optCMAKE='-DINTTYPES_FORMAT:STRING=C99  -DNAMESPACE:STRING=google -DCMAKE_INSTALL_PREFIX=../../Install/gflags'
echo  $CMAKE_CONFIG_GENERATOR 
echo  $optCMAKE 
cmake -G"$CMAKE_CONFIG_GENERATOR" "$optCMAKE" ../../"$RepoSource"
cd ..
echo "************************* $Source_DIR -->debug"
cmake --build $RepoSource  --config debug 
echo "************************* $Source_DIR -->release"
cmake --build $RepoSource  --config release 
cd ..


RepoSource=glog
MAJGitRepo $RepoSource https://github.com/google/glog.git
cd Build/$RepoSource
optCMAKE='-DCMAKE_INSTALL_PREFIX=../../Install/glog'
cmake -G"$CMAKE_CONFIG_GENERATOR" $optCMAKE ../../"$RepoSource" 
cd ..
echo "************************* $Source_DIR -->debug"
cmake --build $RepoSource  --config debug 

echo "************************* $Source_DIR -->release"
cmake --build $RepoSource  --config release 
cmake --build $RepoSource  --target install --config debug 
rename $myRepo/Install/glog/lib/glog.lib $myRepo/Install/glog/lib/glogd.lib
cmake --build $RepoSource  --target install --config release 

cd ..

RepoSource=vtk
MAJGitRepo $RepoSource https://gitlab.kitware.com/vtk/vtk.git
cd Build/$RepoSource
optCMAKE='-DCMAKE_INSTALL_PREFIX=../../Install/vtk -DBUILD_TESTING:BOOL=OFF'
cmake -G"$CMAKE_CONFIG_GENERATOR"  $optCMAKE ../../"$RepoSource"
cd ..
echo "************************* $Source_DIR -->debug"
cmake --build $RepoSource  --config debug 
echo "************************* $Source_DIR -->release"
cmake --build $RepoSource  --config release 
cmake --build $RepoSource  --target install --config release 
cmake --build $RepoSource  --target install --config debug 
cd ..

RepoSource=ceres-solver
MAJGitRepo $RepoSource https://github.com/ceres-solver/ceres-solver.git
cd Build/$RepoSource
optCMAKE='-DEIGEN_INCLUDE_DIR=../../eigen -DCMAKE_INSTALL_PREFIX=../../Install/ceres-solver'
cmake -G"$CMAKE_CONFIG_GENERATOR" $optCMAKE -Dglog_DIR:PATH="$myRepo"/Install/glog -Dgflags_DIR:PATH="$myRepo"/Build/gflags ../../"$RepoSource" 
cd ..
echo "************************* $Source_DIR -->debug"
cmake --build $RepoSource  --config debug 
echo "************************* $Source_DIR -->release"
cmake --build $RepoSource  --config release 
cmake --build $RepoSource  --target install --config release 
cmake --build $RepoSource  --target install --config debug 
cd ..


RepoSource=opencv
cd Build/$RepoSource
optCMAKE=( )

echo $optCMAKE

cmake -G"$CMAKE_CONFIG_GENERATOR" -DBUILD_PERF_TESTS:BOOL=OFF -DBUILD_TESTS:BOOL=OFF -DWITH_DOCS:BOOL=OFF  -DWITH_CUDA:BOOL=OFF -DBUILD_EXAMPLES:BOOL=OFF -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -DCMAKE_INSTALL_PREFIX=../../install/"$RepoSource" -DVTK_DIR="$myRepo"/install/vtk/lib/cmake/vtk-8.1 -Dglog_DIR:PATH="$myRepo"/install/glog/cmake -Dgflags_DIR:PATH="$myRepo"/build/gflags -DCeres_DIR:PATH="$myRepo"/install/ceres-solver/cmake -DEIGEN_DIR:PATH="$myRepo"/eigen -DEIGEN_INCLUDE_PATH=../../eigen ../../"$RepoSource"

cd ..
echo "************************* $Source_DIR -->devenv debug"
cmake --build $RepoSource  --config debug 
echo "************************* $Source_DIR -->devenv release"
cmake --build $RepoSource  --config release 
cmake --build $RepoSource  --target install --config release 
cmake --build $RepoSource  --target install --config debug 
cd ..

