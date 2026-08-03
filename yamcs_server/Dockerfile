# Yamcs ground station for the perseverance rover.
#
# Yamcs publishes no standalone server image, only example images that bundle the server runtime
# with a demo mission. This starts from one of those, strips the demo configuration and mission
# database, and drops in the OmniLRS ones.
#
# Build context is the repository root so that cfg/mdb can be copied in — see docker-compose.yml.
FROM yamcs/example-simulation:5.13.2

USER root

# Remove the bundled demo mission: its instance config, TSE config and mission database.
RUN rm -f /opt/yamcs/etc/yamcs.yaml \
          /opt/yamcs/etc/yamcs.simulator.yaml \
          /opt/yamcs/etc/processor.yaml \
          /opt/yamcs/etc/tse.yaml \
 && rm -rf /opt/yamcs/mdb \
 && mkdir -p /storage/yamcs-data

COPY yamcs_server/etc/ /opt/yamcs/etc/
COPY cfg/mdb/perseverance.xml /opt/yamcs/mdb/perseverance.xml

EXPOSE 8090
