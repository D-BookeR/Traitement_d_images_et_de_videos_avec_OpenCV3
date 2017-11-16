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
myRepo=$(pwd)
CMAKE_CONFIG_GENERATOR="Visual Studio 14 2015 Win64"
mkdir livre
cd livre
RepoSource=OpenCV
cd ..
specificProject1="ChSfm"
specificProject1="Ch7Stereo"
for f in livre/OpenCV/SourceLivreOpenCV/*; do
  if [ -d "$f" ]; then
      echo "*************************  $f"
	Source_DIR="$(basename $f)"
	extproject=".sln"
	mkdir Build/$Source_DIR
	cd Build/$Source_DIR
	echo "$Source_DIR -->CMake"
	if [ "$Source_DIR" = "$specificProject1" ]; then 
	echo "$Source_DIR $specificProject1 -->ceres and vtk"
		cmake -G"$CMAKE_CONFIG_GENERATOR"  -DOpenCV_DIR:PATH="$myRepo"/install/opencv -Dceres_DIR:PATH="$myRepo"/install/ceres-solver/CMake -DVTK_DIR:PATH="$myRepo"/install/vtk/lib/cmake/vtk-8.1  ../../livre/OpenCV/SourceLivreOpencv/$Source_DIR
	else
		if [ "$Source_DIR" = "$specificProject2" ]; then
				cmake -G "Visual Studio 14 2015 Win64"  -DOpenCV_DIR:PATH="$myRepo"/install/opencv -DVTK_DIR:PATH="$myRepo"/install/vtk/lib/cmake/vtk-8.1  ../../livre/OpenCV/SourceLivreOpencv/$Source_DIR
		else
			cmake -G "Visual Studio 14 2015 Win64"  -DOpenCV_DIR:PATH="$myRepo"/install/opencv ../../livre/OpenCV/SourceLivreOpencv/$Source_DIR
		fi
	fi
	project=$Source_DIR$extproject
	cd ..
	echo "************************* $Source_DIR -->devenv debug"
	cmake --build $Source_DIR --target $Source_DIR --config debug 
	echo "************************* $Source_DIR -->devenv release"
	cmake --build $Source_DIR --target $Source_DIR --config release 
	cd ..		
  fi   
done
 