package ru;

import javax.imageio.ImageIO;
import javax.net.ServerSocketFactory;
import javax.net.SocketFactory;
import javax.net.ssl.SSLServerSocketFactory;
import javax.net.ssl.SSLSocketFactory;
import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.net.BindException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Objects;
import java.util.Random;

public class SocketProxyImpl implements SocketProxy {
    private ServerSocketFactory serverSocketFactory;
    private ServerSocket serverSocket;

    private SocketFactory socketFactory;
    private Socket socket;

    private BufferedImage image;
    private String serverInfo;

    private Type type;

    public SocketProxyImpl(String[] args){
        for(int i = 0; i < args.length - 1; i++){
            if(args[i].equals("--type")){
                if(args[i + 1].equals("SSL")){
                    this.type = Type.SSL;
                }
                else if(args[i + 1].equals("Simple")){
                    this.type = Type.SIMPLE;
                }
                else{
                    throw new IllegalArgumentException("Wrong value of --type");
                }
            }
        }
    }

    @Override
    public void sendNext() {
        if(Objects.isNull(socket)){
            throw new IllegalStateException("Connection to the server refused");
        }
        try {
            System.out.println(image.getHeight());
            ImageIO.write(image, "png", socket.getOutputStream());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void accept() {
        setUpSocketFactories();

        try {
            serverSocket = serverSocketFactory.createServerSocket(10000);
        } catch (IOException e) {
            e.printStackTrace();
        }

        try (Socket socket = serverSocket.accept()){
            System.out.println("Waiting for images");
            DataInputStream id = new DataInputStream(socket.getInputStream());
            serverInfo = id.readUTF();
            BufferedInputStream i = new BufferedInputStream(socket.getInputStream());
            image = ImageIO.read(ImageIO.createImageInputStream(i));
            try {
                String[] hostPort = serverInfo.split(";");
                this.socket = socketFactory.createSocket(hostPort[0], Integer.parseInt(hostPort[1]));
            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private void setUpSocketFactories(){
        if(type.equals(Type.SIMPLE)) {
            System.out.println("Simple server is used");
            serverSocketFactory = ServerSocketFactory.getDefault();
            socketFactory = SocketFactory.getDefault();
        }
        else if(type.equals(Type.SSL)){
            System.out.println("SSl server is used");
            serverSocketFactory = SSLServerSocketFactory.getDefault();
            socketFactory = SSLSocketFactory.getDefault();
        }
    }

    @Override
    public void createNoise() {
        Random noise = new Random();
        int temp = noise.nextInt() % 256;
        int p = noise.nextInt() % 101;
        for(int i = 0; i < image.getHeight(); i++){
            for(int j = 0; j < image.getWidth(); j++){
                if(p > 50) {
                    image.setRGB(j, i, temp);
                }
            }
        }
    }

    @Override
    public void cleanup() {
        try {
            serverSocket.close();
            socket.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
