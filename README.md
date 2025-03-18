# CUDA Learning Roadmap & Examples

CUDA 학습을 위한 로드맵과 실습 예제 모음입니다.

## 진행중인 학습 프로젝트

### CUDA C++ Programming Guide 학습

- 2025-03-08 ~ ing

| 섹션 번호 | 섹션 제목                                       | 완료 날짜 | 상태 |
|---------|------------------------------------------------|-----------|------|
| 1       | Introduction                                   | 2025-03-08 | ✅ |
| 2       | Programming Model                              | 2025-03-11 | ✅ |
| 3       | Programming Interface                          |            | 🚧 |
| 4       | Hardware Implementation                        | 2025-03-18 | ✅ |
| 5       | Performance Guidelines                         |            | ❌ |
| 6       | CUDA-Enabled GPUs                              | 2025-03-15 | ✅ |
| 7       | C++ Language Extensions                        |            | 🚧 |
| 8       | Cooperative Groups                             |            | ❌ |
| 9       | CUDA Dynamic Parallelism                       |            | ❌ |
| 10      | Virtual Memory Management                      |            | ❌ |
| 11      | Stream Ordered Memory Allocator                |            | ❌ |
| 12      | Graph Memory Nodes                             |            | ❌ |
| 13      | Mathematical Functions                         |            | ❌ |
| 14      | C++ Language Support                           |            | ❌ |
| 15      | Texture Fetching                               |            | ❌ |
| 16      | Compute Capabilities                           |            | ❌ |
| 17      | Driver API                                     |            | ❌ |
| 18      | CUDA Environment Variables                    |            | ❌ |
| 19      | Unified Memory Programming                     |            | ❌ |
| 20      | Lazy Loading                                   |            | ❌ |
| 21      | Extended GPU Memory                            |            | ❌ |
| 22      | Notices                                        |            | ❌ |


## 🗺️ CUDA 학습 로드맵

### 1. CUDA 기초

- [ ] CUDA 아키텍처 이해
  - GPU 하드웨어 구조
  - CUDA 프로그래밍 모델
- [ ] 기본 개념
  - 스레드/블록/그리드
  - 커널 함수
  - 기본 메모리 모델

### 2. 핵심 개념

- [ ] 메모리 계층구조
  - 글로벌 메모리
  - 공유 메모리
  - 레지스터
  - 상수 메모리
- [ ] 동기화
  - 스레드 동기화
  - 블록 동기화
- [ ] 스트림과 이벤트
  - 비동기 실행
  - 이벤트 기반 동기화

### 3. 성능 최적화

- [ ] 메모리 최적화
  - 메모리 접근 패턴
  - 메모리 정렬
  - 뱅크 충돌 방지
- [ ] 병렬 패턴
  - 리덕션
  - 스캔
  - 히스토그램
- [ ] 성능 분석
  - Nsight 프로파일링
  - 메모리 대역폭 최적화
  - 점유율 최적화

### 4. 고급 주제

- [ ] Multi-GPU 프로그래밍
- [ ] 동적 병렬처리
- [ ] CUDA 라이브러리 활용
  - cuBLAS
  - cuDNN
  - Thrust

## 📚 예제 프로젝트

### Basic

- 간단한 CUDA 커널 실행 예제
- `kernel.cu`: "cuda kernel called!" 메시지를 출력하는 기본 커널 구현
- 10개의 스레드로 커널 실행

### Measurement

- GPU 메모리 정보를 측정하고 출력하는 도구
- 주요 기능:
  - 전체 GPU 메모리 용량 확인
  - 사용 가능한 메모리 용량 확인
  - 메모리 사용량 레벨별(High/Medium/Low) 계산

### Memory

- 비동기 메모리 관리를 위한 `AsyncMemManager` 클래스 구현
- CUDA 스트림을 사용한 비동기 메모리 할당/해제
- 메모리 작업 동기화 기능

### Utils

- 시스템 메모리 계산을 위한 유틸리티 함수
- TensorFlow에서 포팅된 메모리 관리 헬퍼 함수 포함

## 🛠️ 빌드 시스템

프로젝트는 Bazel을 사용하여 빌드됩니다. 각 디렉토리의 `BUILD.bazel` 파일에서 빌드 설정을 확인할 수 있습니다.

### 빌드 요구사항

- CUDA Toolkit
- Bazel
- C++ 컴파일러

### 빌드 방법

```bash
bazel build //...
```

## 📖 학습 자료

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog)

## 📝 라이선스

utils 모듈의 일부 코드는 Apache License 2.0 하에 배포된 TensorFlow 프로젝트에서 가져왔습니다.
