package ru.repository;

import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;
import ru.entities.FileDescriptor;

import java.util.UUID;

@Repository
public interface FileDescriptorRepository extends CrudRepository<FileDescriptor, UUID> {
}
