#export LD_LIBRARY_PATH=../lib:$LD_LIBRARY_PATH
#export ADSP_LIBRARY_PATH="../lib;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/vendor/dsp"

#export ADSP_LIBRARY_PATH=/vendor/lib/rfsa/adsp:/vendor/dsp:$ADSP_LIBRARY_PATH
#export LD_LIBRARY_PATH=../lib2:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=../lib:/vendor/lib:/system/lib:/vendor/lib64:/system/vendor/lib64:$LD_LIBRARY_PATH
#export PATH=/data/local/tmp:$PATH

if [ -z "$1" ]; then
    echo "Usage: $0 <lib_dir>"
    exit 1
fi

LIB_DIR=$1

export VENDOR_LIB=/vendor/lib64/
export LD_LIBRARY_PATH=$LIB_DIR:/vendor/dsp/cdsp:$VENDOR_LIB
export ADSP_LIBRARY_PATH="$LIB_DIR;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
