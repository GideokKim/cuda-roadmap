# CUDA Functions Examples

CUDA 기능들을 테스트하고 실험하기 위한 예제 모음입니다.

## 프로젝트 구조

## 모듈 설명

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

## 빌드 시스템

프로젝트는 Bazel을 사용하여 빌드됩니다. 각 디렉토리의 `BUILD.bazel` 파일에서 빌드 설정을 확인할 수 있습니다.

## 의존성

![image](/img/deps.png)

## 라이선스

utils 모듈의 일부 코드는 Apache License 2.0 하에 배포된 TensorFlow 프로젝트에서 가져왔습니다.
