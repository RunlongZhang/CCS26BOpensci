# Efficient Authentication Function Tree

A C++ research codebase for **function sorting** experiments (e.g., ITree / FsTree variants and Merkle-tree-based proofs). Includes dataset generators and an LP solver target backed by HiGHS Simplex.

> **Build system:** CMake
> **C++ standard:** C++26 (`CMAKE_CXX_STANDARD = 26`)
> **Target platform:** Windows 11 x64

---

## Prebuilt binaries

If you just want to run the programs without building from source, prebuilt Windows x64 executables are available in [`prebuilt/`](prebuilt/). They ship with the runtime DLLs they need and should run on any modern Windows 11 x64 machine without further setup.


To build the project from source instead, follow the instructions below.

---

## Requirements

- **Windows 11 x64**
- **Visual Studio 2022 or later** with the *Desktop development with C++* workload installed (provides MSVC, the Windows SDK, Ninja, and CMake)
- **vcpkg** for installing C++ dependencies (OpenSSL, HiGHS)
- **Eigen** (header-only)

---

## Setup

> **Path placeholders used below:**
> - `<project-root>` — wherever this repository lives on your machine (e.g., `C:\Code`)
> - `<vcpkg-root>` — your vcpkg installation directory (e.g., `C:\vcpkg`)
> - `<eigen-root>` — your local Eigen install directory
>
> Substitute the real paths from your machine when running the commands.

### 1. Install vcpkg

If you don't already have vcpkg, clone and bootstrap it:

```powershell
git clone https://github.com/microsoft/vcpkg.git <vcpkg-root>
<vcpkg-root>\bootstrap-vcpkg.bat
<vcpkg-root>\vcpkg.exe integrate install
```


### 2. Install C++ dependencies

From any PowerShell terminal:

```powershell
<vcpkg-root>\vcpkg.exe install openssl:x64-windows highs:x64-windows
```


### 3. Install Eigen (header-only)

Download or clone Eigen anywhere on disk:

```cmake
include_directories("C:/path/to/your/eigen")
```

---

## Build

### Prerequisite: Edit CMakeLists.txt Paths

1. Edit line 9 to include path to eigen as shown in previous step.
2. Edit line 16 to include path to Highs.h file
3. Edit line 19 to include path to Highs.lib file

### Option A: Visual Studio — recommended

1. Launch Visual Studio.
2. **File → Open → Folder...** and select the project root (`<project-root>`).
3. Visual Studio detects `CMakeLists.txt` and configures automatically. The vcpkg toolchain file is picked up via `CMAKE_TOOLCHAIN_FILE`.
4. In the configuration dropdown at the top of the IDE, select **`x64-Release`**. If it's not in the list, click **Manage Configurations...**, press **+**, and add **x64-Release**.
5. **Build → Build All**.

Executables land in `out\build\x64-Release\` alongside the runtime DLLs.

### Option B: Command line

Open *PowerShell*:

```powershell
cmake -B out/build/x64-Release -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake
cmake --build out/build/x64-Release
cmake --install out/build/x64-Release --prefix dist
```

## Run

From the build output directory (or the prebuilt folder), launch any executable:

YOU MUST GENERATE FUNCTIONS FIRST BEFORE TREES CAN BE BUILT

```powershell
cd out\build\x64-Release
.\FunctionGenerator.exe <functions: int> <dimensions: int>
.\AOF.exe <functions: int> <dimensions: int> <ADS: 0/1>
.\Pruned_AOF.exe <functions: int> <dimensions: int> <ADS: 0/1>
.\Verified_AOF.exe <functions: int> <dimensions: int> <ADS: 0/1>
.\ITreeSimplex.exe <functions: int> <dimensions: int> <ADS: 0/1>
.\format.exe <Itree:0/1> <Pruned_AOF:0/1> <Verified_AOF:0/1> <dimensions: int> <ADS:0/1>
```

Note that format.exe is meant to pool all results together, and should be used when relevant results have been generated.

Alternatively, the .exe in prebuilt can be run with the same parameters in /prebuilt

## Graphing

Graphing code is included as "graphit.py", which requires python and seaborn to work.

```pip install seaborn```

To run:

```python graphit.py <Itree:0/1> <AOF:0/1> <Pruned_AOF:0/1> <Verified_AOF:0/1> <FuncxInters: 0/1> <y-axis: 0/1/2/3> <dimensions: int>```

FuncxInters 0: x-axis represented as n
            1: x-axis represented as n(n-1)/2

y-axis 0: construction time
       1: index storage without ADS
       2: ADS storage without index
       3: total storage

Note that graphit.py needs to be in the same directory as other executables.

If you are using the prebuilt executables, simply move graphit.py to /prebuilt

## Example Sequence

```powershell
cd out\build\x64-Release
.\FunctionGenerator.exe 100 2
.\FunctionGenerator.exe 20 2
.\Verified_AOF.exe 100 2 0
.\Verified_AOF.exe 20 2 0
.\Pruned_AOF.exe 100 2 0
.\Pruned_AOF.exe 20 2 0
.\AOF.exe 100 2 0
.\AOF.exe 20 2 0
.\format.exe 1 0 1 2 0
python graphit.py 0 1 1 1 0 3 2
```

The above sequence: generates 2 data sets at 2 dimensions, constructs all proposed structures, formats the data into one file, and plots the formatted data for total storage
