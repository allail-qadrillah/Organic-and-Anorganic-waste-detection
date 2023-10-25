# Raspberry Pi + MG90S Servo PWM Control Python Code
import RPi.GPIO as GPIO
import time

class Servo:
    def __init__(self, servo_pin:int, pwm_period:int=50):
        self.servo_pin = servo_pin
        self.pwm_period= pwm_period
    
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.servo_pin, GPIO.OUT)
        GPIO.setwarnings(False)

        self.pwm = GPIO.PWM(self.servo_pin, self.pwm_period)
        self.pwm.start(7)

    def rotate_180(self):
        "tottae ser to 180 degrees"
        self.pwm.ChangeDutyCycle(2.0)
        time.sleep(0.5)
        self.pwm.ChangeDutyCycle(12.0)
        time.sleep(0.5)

        GPIO.cleanup()

# servo = Servo( servo_pin=13 )
# servo.rotate_180()
