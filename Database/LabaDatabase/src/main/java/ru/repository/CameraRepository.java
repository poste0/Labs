package ru.repository;

import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;
import ru.entities.Camera;

import java.util.UUID;

@Repository
public interface CameraRepository extends CrudRepository<Camera, UUID> {
}
