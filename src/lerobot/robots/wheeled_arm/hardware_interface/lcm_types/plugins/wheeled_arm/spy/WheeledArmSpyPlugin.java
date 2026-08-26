package wheeled_arm.spy;

import lcm.spy.ChannelData;
import lcm.spy.ChartData;
import lcm.spy.ObjectPanel;
import lcm.spy.SpyPlugin;

import javax.swing.AbstractAction;
import javax.swing.Action;
import javax.swing.BorderFactory;
import javax.swing.JDesktopPane;
import javax.swing.JFrame;
import javax.swing.JInternalFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTable;
import javax.swing.JTextArea;
import javax.swing.Timer;
import javax.swing.SwingUtilities;
import javax.swing.table.AbstractTableModel;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public final class WheeledArmSpyPlugin implements SpyPlugin {
    private static final String[] TYPE_NAMES = {
        "hal.arm.joint_state_t",
        "hal.arm.joint_command_t",
        "hal.head.joint_state_t",
        "hal.head.joint_command_t",
        "hal.waist.joint_state_t",
        "hal.waist.joint_command_t",
        "hal.gripper.gripper_state_t",
        "hal.gripper.gripper_command_t",
        "hal.robot.robot_info_t",
        "hal.canfd.canfd_state_t",
        "hal.canfd.canfd_command_t",
        "hal.leg.joint_state_t",
        "hal.leg.joint_command_t",
        "hal.suction.suction_state_t",
        "hal.suction.suction_command_t",
        "manipulation.arm.joint_state_t",
        "manipulation.arm.joint_command_t",
        "manipulation.head.joint_state_t",
        "manipulation.head.joint_command_t",
        "manipulation.waist.joint_state_t",
        "manipulation.waist.joint_command_t",
        "manipulation.gripper.gripper_state_t",
        "manipulation.gripper.gripper_command_t",
        "manipulation.robot.robot_info_t",
        "manipulation.leg.joint_state_t",
        "manipulation.leg.joint_command_t",
        "manipulation.suction.suction_state_t",
        "manipulation.suction.suction_command_t",
    };

    private static final Set<Long> FINGERPRINTS = loadFingerprints();
    private static final Set<String> FIELD_SKIP_NAMES = new HashSet<String>(
        Arrays.asList("LCM_FINGERPRINT", "LCM_FINGERPRINT_BASE")
    );

    @Override
    public boolean canHandle(long fingerprint) {
        return FINGERPRINTS.contains(Long.valueOf(fingerprint));
    }

    @Override
    public Action getAction(final JDesktopPane desktop, final ChannelData channelData) {
        return new AbstractAction("Wheeled Arm View") {
            @Override
            public void actionPerformed(ActionEvent event) {
                openViewer(desktop, channelData);
            }
        };
    }

    private static Set<Long> loadFingerprints() {
        Set<Long> fingerprints = new HashSet<Long>();
        for (String typeName : TYPE_NAMES) {
            try {
                Class<?> type = Class.forName(typeName);
                Field fingerprint = type.getField("LCM_FINGERPRINT");
                fingerprints.add(Long.valueOf(fingerprint.getLong(null)));
            } catch (ReflectiveOperationException ignored) {
                // A type may be absent when users package a reduced set of messages.
            }
        }
        return fingerprints;
    }

    private static void openViewer(JDesktopPane desktop, ChannelData channelData) {
        openStructureViewer(desktop, channelData);

        WheeledArmPanel panel = new WheeledArmPanel(channelData);
        JInternalFrame frame = new JInternalFrame(
            "Wheeled Arm - " + channelData.name,
            true,
            true,
            true,
            true
        );
        frame.setContentPane(panel);
        frame.setSize(820, 560);
        frame.setMinimumSize(new Dimension(560, 360));
        desktop.add(frame);
        frame.setVisible(true);
        desktop.revalidate();
        desktop.repaint();
        panel.start();
    }

    private static void openStructureViewer(JDesktopPane desktop, final ChannelData channelData) {
        if (channelData.viewerFrame != null && !channelData.viewerFrame.isVisible()) {
            channelData.viewerFrame.dispose();
            channelData.viewer = null;
        }

        if (channelData.viewer == null) {
            long chartStart = channelData.last_utime > 10_000_000L ? channelData.last_utime - 10_000_000L : 0L;
            channelData.viewerFrame = new JFrame(channelData.name);
            channelData.viewer = new ObjectPanel(channelData.name, new ChartData(chartStart));

            JScrollPane viewerScrollPane = new JScrollPane(channelData.viewer);
            viewerScrollPane.getVerticalScrollBar().setUnitIncrement(16);
            channelData.viewer.setViewport(viewerScrollPane.getViewport());

            channelData.viewerFrame.setLayout(new BorderLayout());
            channelData.viewerFrame.add(viewerScrollPane, BorderLayout.CENTER);
            channelData.viewerFrame.setSize(650, 400);
            channelData.viewerFrame.setLocationByPlatform(true);
            channelData.viewerFrame.setVisible(true);
            SwingUtilities.invokeLater(new Runnable() {
                public void run() {
                    channelData.viewer.setObject(channelData.last, channelData.last_utime);
                }
            });
        } else {
            channelData.viewerFrame.setVisible(true);
            channelData.viewerFrame.toFront();
        }
    }

    private static final class WheeledArmPanel extends JPanel {
        private final ChannelData channelData;
        private final FieldTableModel tableModel;
        private final JTextArea summary;
        private final Timer timer;
        private long lastCount = -1;

        WheeledArmPanel(ChannelData channelData) {
            super(new BorderLayout(8, 8));
            this.channelData = channelData;
            this.tableModel = new FieldTableModel();
            this.summary = new JTextArea();
            this.summary.setEditable(false);
            this.summary.setLineWrap(false);
            this.summary.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

            JTable table = new JTable(tableModel);
            table.setAutoCreateRowSorter(true);
            table.getColumnModel().getColumn(0).setPreferredWidth(260);
            table.getColumnModel().getColumn(1).setPreferredWidth(140);
            table.getColumnModel().getColumn(2).setPreferredWidth(360);

            JSplitPane split = new JSplitPane(
                JSplitPane.VERTICAL_SPLIT,
                new JScrollPane(table),
                new JScrollPane(summary)
            );
            split.setResizeWeight(0.72);
            add(split, BorderLayout.CENTER);

            this.timer = new Timer(200, new AbstractAction() {
                @Override
                public void actionPerformed(ActionEvent event) {
                    refresh();
                }
            });
        }

        void start() {
            refresh();
            timer.start();
        }

        private void refresh() {
            if (channelData.nreceived == lastCount) {
                return;
            }
            lastCount = channelData.nreceived;
            Object message = channelData.last;
            tableModel.setRows(rowsFor(message));
            summary.setText(summaryFor(message));
            summary.setCaretPosition(0);
        }

        private String summaryFor(Object message) {
            StringBuilder builder = new StringBuilder();
            builder.append("channel: ").append(channelData.name).append('\n');
            builder.append("type: ").append(channelData.cls == null ? "<unknown>" : channelData.cls.getName()).append('\n');
            builder.append("messages: ").append(channelData.nreceived).append('\n');
            builder.append("errors: ").append(channelData.nerrors).append('\n');
            builder.append("hz: ").append(formatDouble(channelData.hz)).append('\n');
            builder.append("bandwidth: ").append(formatDouble(channelData.bandwidth)).append(" B/s").append('\n');
            if (message == null) {
                builder.append('\n').append("Waiting for the first decoded message...");
            }
            return builder.toString();
        }
    }

    private static final class FieldTableModel extends AbstractTableModel {
        private final String[] columns = {"Field", "Type", "Value"};
        private List<Row> rows = new ArrayList<Row>();

        void setRows(List<Row> rows) {
            this.rows = rows;
            fireTableDataChanged();
        }

        @Override
        public int getRowCount() {
            return rows.size();
        }

        @Override
        public int getColumnCount() {
            return columns.length;
        }

        @Override
        public String getColumnName(int column) {
            return columns[column];
        }

        @Override
        public Object getValueAt(int rowIndex, int columnIndex) {
            Row row = rows.get(rowIndex);
            if (columnIndex == 0) {
                return row.path;
            }
            if (columnIndex == 1) {
                return row.type;
            }
            return row.value;
        }
    }

    private static final class Row {
        final String path;
        final String type;
        final String value;

        Row(String path, String type, String value) {
            this.path = path;
            this.type = type;
            this.value = value;
        }
    }

    private static List<Row> rowsFor(Object message) {
        List<Row> rows = new ArrayList<Row>();
        if (message == null) {
            return rows;
        }
        appendRows(rows, "", message, 0);
        return rows;
    }

    private static void appendRows(List<Row> rows, String path, Object value, int depth) {
        if (value == null) {
            rows.add(new Row(path, "null", ""));
            return;
        }
        Class<?> cls = value.getClass();
        if (isScalar(cls)) {
            rows.add(new Row(path, simpleName(cls), String.valueOf(value)));
            return;
        }
        if (cls.isArray()) {
            int length = Array.getLength(value);
            rows.add(new Row(path, simpleName(cls), "length=" + length));
            int limit = Math.min(length, 32);
            for (int i = 0; i < limit; i++) {
                appendRows(rows, path + "[" + i + "]", Array.get(value, i), depth + 1);
            }
            if (length > limit) {
                rows.add(new Row(path + "[...]", simpleName(cls.getComponentType()), "+" + (length - limit) + " more"));
            }
            return;
        }
        if (depth >= 4) {
            rows.add(new Row(path, simpleName(cls), String.valueOf(value)));
            return;
        }
        for (Field field : cls.getFields()) {
            if (Modifier.isStatic(field.getModifiers()) || FIELD_SKIP_NAMES.contains(field.getName())) {
                continue;
            }
            try {
                String fieldPath = path.length() == 0 ? field.getName() : path + "." + field.getName();
                appendRows(rows, fieldPath, field.get(value), depth + 1);
            } catch (IllegalAccessException ignored) {
                // Public LCM fields should be accessible; skip if a JVM policy says otherwise.
            }
        }
    }

    private static boolean isScalar(Class<?> cls) {
        return cls.isPrimitive()
            || Number.class.isAssignableFrom(cls)
            || Boolean.class.equals(cls)
            || Character.class.equals(cls)
            || String.class.equals(cls);
    }

    private static String simpleName(Class<?> cls) {
        if (cls.isArray()) {
            return simpleName(cls.getComponentType()) + "[]";
        }
        return cls.getSimpleName();
    }

    private static String formatDouble(double value) {
        return String.format("%.2f", Double.valueOf(value));
    }
}
