#pragma once

#ifdef _WIN32
#ifdef __SOYNET__
#define SOYNET_DLL __declspec(dllexport)
#else
#define SOYNET_DLL __declspec(dllimport)
#endif
#else
#define SOYNET_DLL 
#endif

extern "C" {
	SOYNET_DLL void* initSoyNet(const char* cfg_file_name, const char* extend_param); // SoyNet handle¿ª return«—¥Ÿ.
	SOYNET_DLL void feedData(const void * SoyNetHandle, const void* data);
	SOYNET_DLL void feedDataAux(const void * SoyNetHandle, const void* aux);
	SOYNET_DLL void inference(const void * SoyNetHandle);
	SOYNET_DLL void getOutput(const void * SoyNetHandle, void * output);
	SOYNET_DLL void inferSoyNet(const void * soynetHandle, void* input, int input_byte, void* output, int output_byte, char* infer_id, int id_byte, char* extend_param);
	SOYNET_DLL void freeSoyNet(const void* SoyNet);

	//SOYNET_DLL void inference(const void * SoyNetHandle, void * output, const void* data);
}
