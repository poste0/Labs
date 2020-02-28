package ru;

import javax.net.SocketFactory;
import javax.net.ssl.SSLServerSocket;
import javax.net.ssl.SSLSocketFactory;
import java.io.IOException;

public class SSLSocketClientImpl extends AbstractSocketClientImpl {

    public SSLSocketClientImpl(String[] args) {
        super(args);

        socketFactory = SSLSocketFactory.getDefault();
        try {
            socket = socketFactory.createSocket(proxy, Integer.parseInt(proxyPort));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
