package ru;

public interface SocketProxy {
    void sendNext();

    void accept();

    void createNoise();

    void cleanup();
}
