import cv2
from Robot import Robot


def main():
    # robot = Robot("00:1B:10:21:2C:1B")
    robot = Robot("00:1B:10:31:0A:AB")
    camara = cv2.VideoCapture(0)

    try:
        if not camara.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")

        robot.conectar()
        robot.adelante()

        # robot.atras()
        # robot.izquierda()
        # robot.derecha()
        i=1
        while True:
            correcto, imagen = camara.read()
            if i>=1 and i<=3:
                robot.adelante()
            elif i>=4 and i<=7:
                robot.atras()
            elif i>=8 and i<=10:
                robot.derecha()
            elif i>=11 and i<=13:
                robot.izquierda()
            i+=1

            if i>=14:
                i=0
            if not correcto:
                break

            cv2.imshow("Camara", imagen)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if robot.bluetooth_socket is not None:
            robot.parar()
            robot.cerrar()

        camara.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
