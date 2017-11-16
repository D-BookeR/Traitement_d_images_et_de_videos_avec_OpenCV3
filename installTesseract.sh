#!/bin/bash
function MAJGitRepo
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

echo "Installation de leptonica et tesseract"
myRepo=$(pwd)
CMAKE_CONFIG_GENERATOR="Visual Studio 14 2015 Win64"
RepoSource=zlib
mkdir Build/$RepoSource
cd Build/$RepoSource
cmake . -G"Visual Studio 14 2015 Win64" \
-DCMAKE_INSTALL_PREFIX:PATH="$myRepo"/install/zlib -DINSTALL_BIN_DIR:PATH="$myRepo"/install/zlib/bin \
-DINSTALL_INC_DIR:PATH="$myRepo"/install/zlib/include  -DINSTALL_LIB_DIR:PATH="$myRepo"/install/zlib/lib "$myRepo"/zlib
cmake . -G"Visual Studio 14 2015 Win64" \
-DCMAKE_INSTALL_PREFIX:PATH="$myRepo"/install/zlib -DINSTALL_BIN_DIR:PATH="$myRepo"/install/zlib/bin \
-DINSTALL_INC_DIR:PATH="$myRepo"/install/zlib//include  -DINSTALL_LIB_DIR:PATH="$myRepo"/install/zlib/lib "$myRepo"/"$RepoSource"
cmake --build .  --config release 
cmake --build .  --target install --config release 
cmake --build .  --config debug 
cmake --build .  --target install --config debug 
cd ../..
RepoSource=lpng
mkdir Build/$RepoSource
cd Build/$RepoSource
cp "$myRepo"/"$RepoSource"/scripts/pnglibconf.h.prebuilt "$myRepo"/"$RepoSource"/pnglibconf.h
cmake . -G"Visual Studio 14 2015 Win64" \
-DZLIB_INCLUDE_DIR:PATH="$myRepo"/install/zlib/include -DZLIB_LIBRARY_DEBUG:FILE="$myRepo"/install/zlib/lib/zlibstaticd.lib \
-Dld-version-script:BOOL=OFF -DPNG_TESTS:BOOL=OFF -DAWK:STRING= \
-DZLIB_LIBRARY_RELEASE:FILE="$myRepo"/install/zlib/lib/zlibstatic.lib -DCMAKE_INSTALL_PREFIX="$myRepo"/Install/"$RepoSource" "$myRepo"/"$RepoSource"
cmake --build .  --config release 
cmake --build .  --target install --config release 
cmake --build .  --config debug 
cmake --build .  --target install --config debug 
cd ../..
MAJGitRepo leptonica https://github.com/DanBloomberg/leptonica.git
RepoSource=leptonica
pushd Build/$RepoSource
cmake -G"$CMAKE_CONFIG_GENERATOR" -DPNG_LIBRARY:FILEPATH="$myRepo"/install/lpng/lib/libpng16_static.lib -DPNG_PNG_INCLUDE_DIR:PATH="$myRepo"/install/lpng/include \
-DZLIB_LIBRARY:FILEPATH="$myRepo"/install/zlib/lib/zlibstatic.lib -DZLIB_INCLUDE_DIR:PATH="$myRepo"/install/zlib/include \
-DCMAKE_INSTALL_PREFIX="$myRepo"/Install/leptonica "$myRepo/$RepoSource"
echo "************************* $Source_DIR -->debug"
cmake --build .  --config release 
cmake --build .  --target install --config release 
popd

RepoSource=tesseract
MAJGitRepo $RepoSource https://github.com/tesseract-ocr/tesseract.git
pushd Build/$RepoSource
cmake -G"$CMAKE_CONFIG_GENERATOR"  -DBUILD_TRAINING_TOOLS:BOOL=OFF -DCMAKE_INSTALL_PREFIX="$myRepo"/Install/tesseract -DLeptonica_DIR:PATH="$myRepo"/Install/leptonica/cmake -DPKG_CONFIG_EXECUTABLE:BOOL=OFF "$myRepo"/"$RepoSource"
echo "************************* $Source_DIR -->release"
cmake --build . --config release 
cmake --build .  --target install --config release 

popd
exit
RepoSource=opencv
pushd Build/$RepoSource
CMAKE_OPTIONS='-DBUILD_PERF_TESTS:BOOL=OFF -DBUILD_TESTS:BOOL=OFF -DBUILD_DOCS:BOOL=ON -DWITH_CUDA:BOOL=OFF'
cmake -G"$CMAKE_CONFIG_GENERATOR"  \
-DTesseract_INCLUDE_DIR:PATH="${myRepo}"/Install/tesseract/include -DTesseract_LIBRARY="${myRepo}"/Install/tesseract/lib/tesseract400.lib -DLept_LIBRARY="${myRepo}"/Install/leptonica/lib/leptonica-1.74.4.lib \
$CMAKE_OPTIONS -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-DINSTALL_CREATE_DISTRIB=ON -DCMAKE_INSTALL_PREFIX="$myRepo"/install/"$RepoSource"  "$myRepo/$RepoSource"
echo "************************* $Source_DIR -->devenv debug"
cmake --build .  --config debug 
echo "************************* $Source_DIR -->devenv release"
cmake --build .  --config release 
cmake --build .  --target install --config release 
cmake --build .  --target install --config debug 
popd

