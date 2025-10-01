register(
    id='contact_rl/contact-v0',
    entry_point='env:ContactRL',
)

register(
    id='contact_lag/contact-v1',
    entry_point='env:ContactLag',
)

register(
    id='contact_cpo/contact-v2',
    entry_point='env:ContactCPO',
)
