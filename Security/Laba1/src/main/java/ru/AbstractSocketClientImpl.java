package ru;

import javax.imageio.ImageIO;
import javax.net.SocketFactory;
import java.awt.image.RenderedImage;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.net.Socket;

public abstract class AbstractSocketClientImpl implements SocketClient {
    protected String proxy = "";
    protected String proxyPort = "";
    protected String server = "";
    protected String serverPort = "";
    protected String imageFile = "";
    protected RenderedImage image = null;
    protected SocketFactory socketFactory;
    protected Socket socket;

    public AbstractSocketClientImpl(String[] args){
        try {
            proxy = getParam(args, "--proxy");
            proxyPort = getParam(args, "--proxyPort");
            server = getParam(args, "--server");
            serverPort = getParam(args, "--serverPort");
            imageFile = getParam(args, "--image");

            image = ImageIO.read(new File(imageFile));
        }
        catch (IllegalArgumentException e){
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void sendServerInfo() {
        try {
            DataOutputStream dataOs = new DataOutputStream(socket.getOutputStream());
            dataOs.writeUTF(server + ";" + serverPort);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void sendImage() {
        try {
            ImageIO.write(image, "png", socket.getOutputStream());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
