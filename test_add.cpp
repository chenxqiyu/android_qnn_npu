#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "QNN/QnnInterface.h"
#include "QNN/QnnBackend.h"
#include "QNN/QnnDevice.h"
#include "QNN/QnnContext.h"
#include "QNN/QnnGraph.h"
#include "QNN/QnnTensor.h"
#include "QNN/QnnLog.h"
#include "QNN/QnnCommon.h"
#include "QNN/QnnTypes.h"
#include "QNN/QnnOpDef.h"

#define QNN_BACKEND_ID_HTP 6
#define BACKEND_ID QNN_BACKEND_ID_HTP
#define PACKAGE_NAME "qti.aisw"

static const QnnInterface_t* g_iface = NULL;
static const QNN_INTERFACE_VER_TYPE* g_ifv = NULL;

static void check_error(Qnn_ErrorHandle_t err, const char* msg) {
    if (err != QNN_SUCCESS) {
        const char* briefMsg = NULL;
        const char* verboseMsg = NULL;
        if (g_ifv && g_ifv->errorGetMessage) g_ifv->errorGetMessage(err, &briefMsg);
        if (g_ifv && g_ifv->errorGetVerboseMessage) g_ifv->errorGetVerboseMessage(err, &verboseMsg);
        fprintf(stderr, "[ERROR] %s failed:\n", msg);
        fprintf(stderr, "  Error code: 0x%llx\n", (unsigned long long)err);
        if (briefMsg) fprintf(stderr, "  Brief message: %s\n", briefMsg);
        if (verboseMsg) {
            fprintf(stderr, "  Verbose message: %s\n", verboseMsg);
            if (g_ifv && g_ifv->errorFreeVerboseMessage) g_ifv->errorFreeVerboseMessage(verboseMsg);
        }
        exit(1);
    }
}

static const QnnInterface_t* get_iface(uint32_t numProviders, const QnnInterface_t** providers) {
    for (uint32_t i = 0; i < numProviders; i++)
        if (providers[i]->backendId == BACKEND_ID) return providers[i];
    return NULL;
}

static Qnn_Tensor_t makeTensor(const char* name,
                               Qnn_TensorType_t type,
                               Qnn_DataType_t dataType,
                               uint32_t* dims,
                               uint32_t rank,
                               void* data,
                               uint32_t dataSize) {
    Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
    tensor.version      = QNN_TENSOR_VERSION_1;
    tensor.v1.name      = name;
    tensor.v1.type      = type;
    tensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    tensor.v1.dataType   = dataType;
    tensor.v1.rank       = rank;
    tensor.v1.dimensions = dims;
    tensor.v1.memType    = QNN_TENSORMEMTYPE_RAW;
    tensor.v1.clientBuf.data = data;
    tensor.v1.clientBuf.dataSize = dataSize;
    return tensor;
}

int main() {
    Qnn_ErrorHandle_t err;
    Qnn_LogHandle_t logger = NULL;
    const QnnInterface_t** providers = NULL;
    uint32_t numProviders = 0;

    err = QnnInterface_getProviders(&providers, &numProviders);
    check_error(err, "getProviders");

    const QnnInterface_t* iface = get_iface(numProviders, providers);
    if (!iface) { fprintf(stderr, "HTP backend not found\n"); return 1; }
    g_ifv = &iface->QNN_INTERFACE_VER_NAME;

    const QNN_INTERFACE_VER_TYPE& ifv = iface->QNN_INTERFACE_VER_NAME;

    err = ifv.logCreate(NULL, QNN_LOG_LEVEL_INFO, &logger);
    check_error(err, "logCreate");

    Qnn_BackendHandle_t backend = NULL;
    err = ifv.backendCreate(logger, NULL, &backend);
    check_error(err, "backendCreate");

    Qnn_DeviceHandle_t device = NULL;
    err = ifv.deviceCreate(logger, NULL, &device);
    check_error(err, "deviceCreate");

    Qnn_ContextHandle_t context = NULL;
    err = ifv.contextCreate(backend, device, NULL, &context);
    check_error(err, "contextCreate");

    printf("[TEST_ADD] Starting tensor addition test...\n");

    Qnn_GraphHandle_t graph = NULL;
    err = ifv.graphCreate(context, "add_graph", NULL, &graph);
    check_error(err, "graphCreate");

    uint32_t dims[] = {4};
    uint32_t numDim = 1;

    // APP_WRITE = app writes data = graph input
    Qnn_Tensor_t tensorA = makeTensor("tensor_A", QNN_TENSOR_TYPE_APP_WRITE,
                                      QNN_DATATYPE_FLOAT_32, dims, numDim, NULL, 0);
    err = ifv.tensorCreateGraphTensor(graph, &tensorA);
    check_error(err, "create tensorA");

    Qnn_Tensor_t tensorB = makeTensor("tensor_B", QNN_TENSOR_TYPE_APP_WRITE,
                                      QNN_DATATYPE_FLOAT_32, dims, numDim, NULL, 0);
    err = ifv.tensorCreateGraphTensor(graph, &tensorB);
    check_error(err, "create tensorB");

    // APP_READ = app reads data = graph output
    Qnn_Tensor_t tensorC = makeTensor("tensor_C", QNN_TENSOR_TYPE_APP_READ,
                                      QNN_DATATYPE_FLOAT_32, dims, numDim, NULL, 0);
    err = ifv.tensorCreateGraphTensor(graph, &tensorC);
    check_error(err, "create tensorC");

    // Configure ElementWiseBinary op
    Qnn_Scalar_t opScalar;
    memset(&opScalar, 0, sizeof(opScalar));
    opScalar.dataType = QNN_DATATYPE_UINT_32;
    opScalar.uint32Value = QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD;

    Qnn_Param_t param;
    memset(&param, 0, sizeof(param));
    param.paramType = QNN_PARAMTYPE_SCALAR;
    param.name = QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION;
    param.scalarParam = opScalar;

    Qnn_Tensor_t inputTensors[2] = {tensorA, tensorB};

    Qnn_OpConfig_t opConfig = QNN_OPCONFIG_INIT;
    opConfig.v1.name = "add_node";
    opConfig.v1.packageName = PACKAGE_NAME;
    opConfig.v1.typeName = QNN_OP_ELEMENT_WISE_BINARY;
    opConfig.v1.numOfParams = 1;
    opConfig.v1.params = &param;
    opConfig.v1.numOfInputs = 2;
    opConfig.v1.inputTensors = inputTensors;
    opConfig.v1.numOfOutputs = 1;
    opConfig.v1.outputTensors = &tensorC;

    err = ifv.graphAddNode(graph, opConfig);
    check_error(err, "graphAddNode");

    err = ifv.graphFinalize(graph, NULL, NULL);
    check_error(err, "graphFinalize");
    printf("[TEST_ADD] Graph created and finalized successfully.\n");

    // Execute with actual data - must use same IDs as registered tensors
    float dataA[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dataB[] = {10.0f, 20.0f, 30.0f, 40.0f};
    float dataC[4] = {0};

    // Copy registered tensors and attach data buffers
    Qnn_Tensor_t execInputA = tensorA;
    execInputA.v1.clientBuf.data = dataA;
    execInputA.v1.clientBuf.dataSize = sizeof(dataA);

    Qnn_Tensor_t execInputB = tensorB;
    execInputB.v1.clientBuf.data = dataB;
    execInputB.v1.clientBuf.dataSize = sizeof(dataB);

    Qnn_Tensor_t execOutputC = tensorC;
    execOutputC.v1.clientBuf.data = dataC;
    execOutputC.v1.clientBuf.dataSize = sizeof(dataC);

    const Qnn_Tensor_t inputs[2] = {execInputA, execInputB};
    Qnn_Tensor_t outputs[1] = {execOutputC};

    err = ifv.graphExecute(graph, inputs, 2, outputs, 1, NULL, NULL);
    check_error(err, "graphExecute");

    printf("[TEST_ADD] Result: C = A + B\n");
    printf("[TEST_ADD] A = [%.1f, %.1f, %.1f, %.1f]\n", dataA[0], dataA[1], dataA[2], dataA[3]);
    printf("[TEST_ADD] B = [%.1f, %.1f, %.1f, %.1f]\n", dataB[0], dataB[1], dataB[2], dataB[3]);
    printf("[TEST_ADD] C = [%.1f, %.1f, %.1f, %.1f]\n", dataC[0], dataC[1], dataC[2], dataC[3]);

    ifv.contextFree(context, NULL);
    ifv.deviceFree(device);
    ifv.backendFree(backend);
    ifv.logFree(logger);

    printf("[TEST_ADD] Test completed successfully.\n");
    return 0;
}
