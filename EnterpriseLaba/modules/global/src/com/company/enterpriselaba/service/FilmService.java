package com.company.enterpriselaba.service;

import com.company.enterpriselaba.entity.*;

import java.sql.Time;
import java.util.Date;

public interface FilmService {
    String NAME = "enterpriselaba_FilmService";

    String getFilmInfo(Film film);

    void addFilm(String name, Date startShowDate, Integer periodOfShowing, String description, Admin admin);

    void changeFilm(Film film, String name, Date startShowDate, Integer periodOfShowing, String description, Admin admin);

    void deleteFilm(Film film);

    void addShow(String name, Film film, Theatre theatre, Auditorium auditorium, Date showDate, Time showTime, Double price, Admin admin);

    void deleteShow(Show show, Admin admin);

    void changeShow(Show show, String name, Film film, Theatre theatre, Auditorium auditorium, Date showDate, Time showTime, Double price, Admin admin);
}