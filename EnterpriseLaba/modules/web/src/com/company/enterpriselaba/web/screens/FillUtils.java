package com.company.enterpriselaba.web.screens;

import com.company.enterpriselaba.entity.Auditorium;
import com.company.enterpriselaba.entity.Film;
import com.company.enterpriselaba.entity.Show;
import com.company.enterpriselaba.entity.Theatre;
import com.haulmont.cuba.core.global.LoadContext;
import com.haulmont.cuba.gui.components.LookupField;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class FillUtils {
    public static void fillFilmField(List<Film> films, LookupField<Film> filmField){
        Map<String, Film> filmMap  = new HashMap<>();

        films.forEach(film -> {
            filmMap.put(film.getName(), film);
        });

        filmField.setOptionsMap(filmMap);
    }

    public static void fillTheatreField(List<Theatre> theatres, LookupField<Theatre> theatreField){
        Map<String, Theatre> theatreMap  = new HashMap<>();

        theatres.forEach(theatre -> {
            theatreMap.put(theatre.getName(), theatre);
        });

        theatreField.setOptionsMap(theatreMap);
    }

    public static void fillAuditoriumField(List<Auditorium> auditoriums, LookupField<Auditorium> auditoriumField){
        Map<String, Auditorium> auditoriumMap  = new HashMap<>();

        auditoriums.forEach(auditorium -> {
            auditoriumMap.put(auditorium.getTheatre().getName() + "'s auditorium " + auditorium.getNumber(), auditorium);
        });

        auditoriumField.setOptionsMap(auditoriumMap);
    }

    public static void fillTypeField(LookupField<String> typeField){
        typeField.setOptionsList(Arrays.asList("Admin", "Employee"));
    }

    public static void fillShowField(List<Show> shows, LookupField<Show> showField){
        Map<String, Show> showMap  = new HashMap<>();

        shows.forEach(show -> {
            showMap.put(show.getName(), show);
        });

        showField.setOptionsMap(showMap);
    }
}
