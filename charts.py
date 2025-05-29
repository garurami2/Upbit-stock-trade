import traceback
import time
import os
import io

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image


def capture_full_page(url, output_path):
    """업비트 차트 페이지의 전체 화면을 캡처하는 함수"""

    # Chrome 옵션 설정
    chrome_options = Options()
    chrome_options.add_argument("--headless") # 헤드리스 모드
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--enable-webgl") # WebGL 명시적 활성화
    chrome_options.add_argument("--disable-software-rasterizer") # software-rasterizer 비활성화
    # chrome_options.add_argument("--disable-gpu")
    # chrome_options.add_argument("--window-size=1920,1080")  # 창 크기 설정


    # WebDriver 설정
    # 웹드라이버 초기화
    driver = webdriver.Chrome(options=chrome_options)

    # 페이지가 완전히 로드될 때까지 대기
    wait = WebDriverWait(driver, 20)

    try:

        # 페이지 로드
        print(f"페이지 로딩 중: {url}")

        # 시간 설정을 1시간으로 변경
        try:
            driver.get(url)

            # 추가 로딩 시간 (차트 데이터 로딩을 위해)
            time.sleep(5)

            print("시간 설정 변경 중...")

            # 시간 버튼 클릭 (메뉴 열기)
            time_button_xpath = "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]/span/cq-clickable"
            time_button = wait.until(EC.element_to_be_clickable((By.XPATH, time_button_xpath)))
            time_button.click()
            print("시간 메뉴 열기 완료")

            # 잠시 대기 (메뉴가 나타날 때까지)
            time.sleep(1)

            # 1시간 옵션 클릭
            hour_option_xpath = "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]/cq-menu-dropdown/cq-item[8]"
            hour_option = wait.until(EC.element_to_be_clickable((By.XPATH, hour_option_xpath)))
            hour_option.click()
            print("1시간 옵션 선택 완료")

            # 차트 업데이트 대기
            time.sleep(3)
            print("차트 업데이트 완료")

        except Exception as e:
            print(f"시간 설정 변경 중 오류 발생: {str(e)}")
            print("기본 설정으로 캡처를 진행합니다...")

        # 볼린저 밴드 지표 추가
        try:
            print("볼린저 밴드 지표 추가 중...")

            # 지표 버튼 클릭 (메뉴 열기)
            indicator_button_xpath = "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/span"
            indicator_button = wait.until(EC.element_to_be_clickable((By.XPATH, indicator_button_xpath)))
            indicator_button.click()
            print("지표 메뉴 열기 완료")

            # 잠시 대기 (메뉴가 나타날 때까지)
            time.sleep(1)

            # 볼린저 밴드 옵션 클릭
            bollinger_option_xpath = "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[15]"
            bollinger_option = wait.until(EC.element_to_be_clickable((By.XPATH, bollinger_option_xpath)))
            bollinger_option.click()
            print("볼린저 밴드 지표 추가 완료")

            # 지표 적용 대기
            time.sleep(3)
            print("지표 적용 완료")

        except Exception as e:
            print(f"볼린저 밴드 지표 추가 중 오류 발생: {str(e)}")
            traceback.print_exc()
            print("지표 없이 캡처를 진행합니다...")

        # 전체 페이지 높이 가져오기
        total_height = driver.execute_script("return document.body.scrollHeight")
        print(f"전체 페이지 높이: {total_height}px")

        # 뷰포트 높이 설정
        driver.set_window_size(1920, total_height)

        # 스크린샷 저장 디렉토리 생성
        if not os.path.exists("screenshots"):
            os.makedirs("screenshots")

        # 스크린샷 캡처
        png = driver.get_screenshot_as_png()

        # PIL Image로 변환
        img = Image.open(io.BytesIO(png))

        # 이미지 리사이즈(OpenAI API 제한에 맞춤)
        img.thumbnail((2000, 2000))

        # 최적화된 이미지 저장
        img.save(output_path, optimize=True, quality=85)
        print(f"Optimized screenshot saved as: {output_path}")

        return True

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return False

    finally:
        # 웹드라이버 종료
        if 'driver' in locals():
            driver.quit()
            print("웹드라이버 종료 완료")