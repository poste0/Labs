package ru;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.entities.*;
import ru.repository.*;

import java.util.*;

@Service
public class DataService {
    @Autowired
    private CameraRepository cameraRepository;

    @Autowired
    private FileDescriptorRepository fileDescriptorRepository;

    @Autowired
    private NodeRepository nodeRepository;

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private VideoRepository videoRepository;

    private final List<String> firstNames = Arrays.asList("Sergei", "Ivan", "Nikolai", "Alexandr", "Dmitriy", "Pavel",
            "Vasiliy", "Darya", "Mariya", "Alena", "Artem", "Olga", "Mikhail",
            "Alexei", "Andrei", "Marina", "Viktor", "Denis", "Danila", "Vadim");

    private final List<String> lastNames = Arrays.asList("Ivanov", "Petrov", "Sidorov", "Alexeev", "Pavlov", "Agutin", "Pushkin", "Lermontov", "Mamontov", "Mendeleev");

    private final List<String> emailDomains = Arrays.asList("gmail.com", "yandex.ru", "ssau.ru", "mail.ru", "some.com");

    private List<User> users = new LinkedList<>();

    private final List<List<Integer>> sizes = Arrays.asList(Arrays.asList(1280, 1024), Arrays.asList(1024, 760));

    private final List<Integer> frameRates = Arrays.asList(25, 30, 50, 100);

    private List<Camera> cameras = new LinkedList<>();

    private final List<String> statuses = Arrays.asList("Processing", "Processed", "Recording", "Stored");

    private final List<String> extensions = Arrays.asList(".mp4", ".avi", ".m3u8", ".flv");

    private final List<String> cpus = Arrays.asList("intel", "amd");

    private final List<String> gpus = Arrays.asList("NVIDIA", "AMD");

    public void fill(){
        Random random = new Random();
        System.out.println("Filling has started");
        firstNames.forEach(firstName -> {
            lastNames.forEach(lastName -> {
                emailDomains.forEach(emailDomain -> {
                    User user = new User();
                    user.setFirstName(firstName);
                    user.setLastName(lastName);

                    String email = "";

                    int subFirstName = random.nextInt(firstName.length());
                    int subLastName = random.nextInt(lastName.length());
                    email = firstName.substring(subFirstName) + lastName.substring(subLastName) + "@" + emailDomain;
                    user.setEmail(email);

                    String login = random(10);

                    String password = random(10);
                    user.setLogin(login);
                    user.setPassword(password);

                    users.add(user);

                    userRepository.save(user);


                });
            });
        });

        users.forEach(user -> {
            List<Camera> camerasForUser = new LinkedList<>();
            for(int i = 0; i < 10; i++) {
                Camera camera = new Camera();
                String name = user.getFirstName() + "_Camera" + camerasForUser.size();
                String address = "rtsp://" + random(10);
                int sizeIndex = random.nextInt(sizes.size());
                Integer heigth = sizes.get(sizeIndex).get(0);
                Integer width = sizes.get(sizeIndex).get(1);
                int frameRateIndex = random.nextInt(frameRates.size());
                Integer frameRate = frameRates.get(frameRateIndex);

                camera.setName(name);
                camera.setAddress(address);
                camera.setWidth(width);
                camera.setHeight(heigth);
                camera.setFrameRate(frameRate);
                camerasForUser.add(camera);
                cameras.add(camera);
                camera.setUser(user);

                cameraRepository.save(camera);
            }
        });

        cameras.forEach(camera -> {
            for(int i = 0; i < 100; i++){
                Video video = new Video();
                String name = random(5);
                int statusIndex = random.nextInt(statuses.size());
                String status = statuses.get(statusIndex);

                FileDescriptor descriptor = new FileDescriptor();
                String descriptorName = random(20);
                int extensionIndex = random.nextInt(extensions.size());
                String extension = extensions.get(extensionIndex);
                Double size = random.nextDouble();
                Date createDate = new Date();

                descriptor.setName(descriptorName);
                descriptor.setExtension(extension);
                descriptor.setSize(size);
                descriptor.setCreateDate(createDate);

                fileDescriptorRepository.save(descriptor);

                video.setName(name);
                video.setStatus(status);
                video.setCamera(camera);
                video.setFileDescriptor(descriptor);

                videoRepository.save(video);
            }
        });
        System.out.println("Filling's over");
    }

    public void fillNodes(){
        List<User> users = userRepository.findAll();
        List<Node> nodes = new LinkedList<>();
        Random random = new Random();

        for(int i = 0; i < 100; i++){
            Node node = new Node();
            String name = random(5);
            String address = "http://" + random(10);
            int cpuIndex = random.nextInt(cpus.size());
            String cpu = cpus.get(cpuIndex);
            int gpuIndex = random.nextInt(gpus.size());
            String gpu = gpus.get(gpuIndex);

            node.setAddress(address);
            node.setName(name);
            node.setCpu(cpu);
            node.setGpu(gpu);

            nodes.add(node);
            nodeRepository.save(node);
        }

        for(int i = 0; i < 10; i++){
            int finalI = i;
            users.forEach(user -> {
                int start = finalI * 20 % 100;
                int end = (finalI + 1) * 20 % 100;
                user.setNodes(nodes.subList(start, end));
                userRepository.save(user);
            });
        }


    }

    public void fillFileDescriptors(){
        Random random = new Random();
        List<Video> videos = videoRepository.findAll();

        videos.forEach(video -> {
            FileDescriptor fileDescriptor = new FileDescriptor();

            String descriptorName = random(20);
            int extensionIndex = random.nextInt(extensions.size());
            String extension = extensions.get(extensionIndex);
            Double size = random.nextDouble();
            Date createDate = new Date();

            fileDescriptor.setCreateDate(createDate);
            fileDescriptor.setSize(size);
            fileDescriptor.setExtension(extension);
            fileDescriptor.setName(descriptorName);

            fileDescriptor.setVideo(video);

            fileDescriptorRepository.save(fileDescriptor);

            video.setFileDescriptor(fileDescriptor);
            videoRepository.save(video);

        });
    }
    private String random(int count){
        int leftLimit = 97; // letter 'a'
        int rightLimit = 122; // letter 'z'
        Random random = new Random();

        String value = random.ints(leftLimit, rightLimit + 1)
                .limit(count)
                .collect(StringBuilder::new, StringBuilder::appendCodePoint, StringBuilder::append)
                .toString();

        return value;
    }
}
