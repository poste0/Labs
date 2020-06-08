package com.company.enterpriselaba.web.screens.theatre;

import com.haulmont.cuba.gui.screen.*;
import com.company.enterpriselaba.entity.Theatre;

@UiController("enterpriselaba_Theatre.browse")
@UiDescriptor("theatre-browse.xml")
@LookupComponent("theatresTable")
@LoadDataBeforeShow
public class TheatreBrowse extends StandardLookup<Theatre> {
}