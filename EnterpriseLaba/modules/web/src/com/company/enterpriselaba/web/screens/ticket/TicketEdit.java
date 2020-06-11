package com.company.enterpriselaba.web.screens.ticket;

import com.company.enterpriselaba.entity.Show;
import com.company.enterpriselaba.service.TicketService;
import com.company.enterpriselaba.web.screens.FillUtils;
import com.haulmont.cuba.core.global.*;
import com.haulmont.cuba.gui.components.*;
import com.haulmont.cuba.gui.screen.*;
import com.company.enterpriselaba.entity.Ticket;
import com.haulmont.cuba.security.entity.User;

import javax.inject.Inject;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;

@UiController("enterpriselaba_Ticket.edit")
@UiDescriptor("ticket-edit.xml")
@EditedEntityContainer("ticketDc")
@LoadDataBeforeShow
public class TicketEdit extends StandardEditor<Ticket> {
    @Inject
    private DataManager dataManager;

    @Inject
    private LookupField<Show> showField;

    @Inject
    private TicketService ticketService;

    @Inject
    private Metadata metadata;

    @Inject
    private TextField<Integer> countOFTickets;

    @Inject
    private Button ticketButton;

    @Inject
    private Label<String> cannotSellLabel;

    @Subscribe
    private void onInit(InitEvent event){
        List<Show> shows = dataManager.loadList(LoadContext.create(Show.class).setQuery(LoadContext.createQuery("SELECT s FROM enterpriselaba_Show s")).setView("show-view"));
        FillUtils.fillShowField(shows, showField);

        showField.addValueChangeListener(new Consumer<HasValue.ValueChangeEvent<Show>>() {
            @Override
            public void accept(HasValue.ValueChangeEvent<Show> showValueChangeEvent) {
                boolean condition = !Objects.isNull(countOFTickets.getValue()) && showValueChangeEvent.getValue().getAuditorium().getCountOfSeats() - showValueChangeEvent.getValue().getTickets().size() < countOFTickets.getValue();
                changeFiledValues(condition);
            }
        });

        countOFTickets.addValueChangeListener(new Consumer<HasValue.ValueChangeEvent<Integer>>() {
            @Override
            public void accept(HasValue.ValueChangeEvent<Integer> integerValueChangeEvent) {
                boolean condition = !Objects.isNull(showField.getValue()) && showField.getValue().getAuditorium().getCountOfSeats() - showField.getValue().getTickets().size() < integerValueChangeEvent.getValue();
                changeFiledValues(condition);
            }
        });
    }

    private void changeFiledValues(boolean condition){
        if(condition){
            ticketButton.setEnabled(false);
            cannotSellLabel.setVisible(true);
        }
        else{
            ticketButton.setEnabled(true);
            cannotSellLabel.setVisible(false);
        }
    }

    public void onClick() {
        if(!Objects.isNull(countOFTickets.getValue())) {
            for (int i = 0; i < countOFTickets.getValue(); i++) {
                Ticket ticket = metadata.create(Ticket.class);
                ticket.setShow(showField.getValue());
                User user = AppBeans.get(UserSessionSource.class).getUserSession().getUser();

                ticketService.buyTicket(user, ticket);
            }
        }

        close(WINDOW_CLOSE_ACTION);
    }
}