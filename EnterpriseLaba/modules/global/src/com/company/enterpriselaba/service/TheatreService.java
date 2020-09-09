package com.company.enterpriselaba.service;

import com.company.enterpriselaba.entity.Admin;
import com.company.enterpriselaba.entity.Auditorium;
import com.company.enterpriselaba.entity.Theatre;

import java.util.List;
import java.util.UUID;

public interface TheatreService {
    String NAME = "enterpriselaba_TheatreService";

    void addTheatre(String name, String address, Admin admin);

    void changeTheatre(Theatre theatre, String name, String address, Admin admin);

    void deleteTheatre(Theatre theatre, Admin admin);

    Theatre getTheatre(UUID id);

    void addAuditorium(Integer countOfSeats, Theatre theatre, Admin admin);

    void changeAuditorium(Auditorium auditorium, Integer countOfSeats, Admin admin);
}